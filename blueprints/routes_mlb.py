from flask import Blueprint, render_template, request, current_app
import pandas as pd
import pulp
import random
import re
import time
from datetime import datetime
import pytz
from thefuzz import process, fuzz
from helpers.mlb_helpers import (
    get_weighted_stats,
    get_mlb_weather_data,
    get_espn_game_info,
    get_prime_matchup,
    get_logo_url,
    get_player_headshot_url,
    TEAM_MAP
)

mlb_bp = Blueprint('mlb', __name__, url_prefix='/mlb')

SALARY_CSV = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRzCRSTDnslz-zmGESH1CFhjsYD7NJa8yHkapMFu1JIR0M1PQDwZzMIDCmhPBUNU6kzLJy8-3_ioR4Y/pub?gid=1189680617&single=true&output=csv"
SLOTS = ['P1', 'P2', 'C', '1B', '2B', '3B', 'SS', 'OF1', 'OF2', 'OF3']
POS_ORDER = {s: i for i, s in enumerate(SLOTS)}


def get_df_raw():
    try:
        sync_url = f"{SALARY_CSV}&t={int(time.time())}"
        df = pd.read_csv(sync_url)
        df.columns = df.columns.str.strip()

        # Modernized fillna logic to prevent FutureWarnings
        for col, target in [('salary', 'Salary'), ('proj', 'Proj_Base')]:
            found = next((c for c in df.columns if col in c.lower()), None)
            if found:
                df[target] = pd.to_numeric(df[found].astype(str).replace(r'[^0-9.]', '', regex=True), errors='coerce')
                df[target] = df[target].fillna(0.0)
            else:
                df[target] = 0.0

        if 0 < df['Salary'].max() < 1000: df['Salary'] *= 1000

        # Set a floor for projected players
        df.loc[(df['Proj_Base'] <= 0.1) & (df['Salary'] > 0), 'Proj_Base'] = 5.0

        # Handle Batting Order
        order_col = next((c for c in df.columns if any(x in c.lower() for x in ['order', 'batting', 'bat  ✓'])), None)
        if order_col:
            df['Order'] = df[order_col].astype(str).str.extract('(\d+)').fillna(0).astype(int)
        else:
            df['Order'] = 0

        df['POS'] = df['POS'].astype(str).str.upper().fillna('UTIL')
        return df
    except Exception as e:
        print(f"MLB CSV ERROR: {e}")
        return pd.DataFrame()


def run_optimizer(df_input, num_lineups=1, locks=[], stack_team=None, min_stack=5, diversity=4, excluded_games=[]):
    df = df_input.copy()

    # 1. Exclusion Logic
    if excluded_games:
        df = df[~df.apply(lambda r: " vs ".join(sorted([TEAM_MAP.get(str(r['Team']), str(r['Team'])),
                                                        TEAM_MAP.get(str(r['Opponent']),
                                                                     str(r['Opponent']))])) in excluded_games, axis=1)]

    all_results, used_player_indices = [], []
    confirmed_teams = set(df[df['Order'] > 0]['Team'].unique())

    # 2. Daily Fantasy Logic (Order & Platoon bonuses)
    def apply_logic(row):
        proj = float(row['Proj_Base'])
        team, order = str(row['Team']), int(row.get('Order', 0))

        if 'P' in str(row['POS']):
            proj += (float(row.get('Chalk_Quality', 0)) * 0.5)
        else:
            if team in confirmed_teams:
                if order == 0: return 0.0  # Player isn't in confirmed lineup
                proj *= (1.20 if order <= 2 else 1.10 if order <= 5 else 0.95)

        return max(proj * random.uniform(0.97, 1.03), 0.1)

    df['Solver_Proj'] = df.apply(apply_logic, axis=1)
    eligible = df[df['Solver_Proj'] > 0.5].copy()
    if len(eligible) < 10: return []

    # 3. Solver Loop
    stack_candidates = [stack_team] if (stack_team and stack_team != "None") else \
        eligible[~eligible['POS'].str.contains('P')].groupby('Team')['Solver_Proj'].mean().sort_values(
            ascending=False).head(8).index.tolist()

    for i in range(num_lineups):
        best_lineup, highest_score = None, -1
        random.shuffle(stack_candidates)

        for current_team in stack_candidates[:3]:
            prob = pulp.LpProblem(f"MLB_{i}_{current_team}", pulp.LpMaximize)
            indices = eligible.index.tolist()
            x = pulp.LpVariable.dicts("x", (indices, SLOTS), cat="Binary")

            prob += pulp.lpSum([eligible.loc[p, 'Solver_Proj'] * x[p][s] for p in indices for s in SLOTS])
            prob += pulp.lpSum([eligible.loc[p, 'Salary'] * x[p][s] for p in indices for s in SLOTS]) <= 50000

            for s in SLOTS: prob += pulp.lpSum([x[p][s] for p in indices]) == 1
            for p in indices:
                prob += pulp.lpSum([x[p][s] for s in SLOTS]) <= 1
                if eligible.loc[p, 'Player'] in locks: prob += pulp.lpSum([x[p][s] for s in SLOTS]) == 1

                # Specific MLB Slot Matching
                pos = str(eligible.loc[p, 'POS'])
                for s in SLOTS:
                    valid = (s.startswith('P') and 'P' in pos) or \
                            (s == 'C' and 'C' in pos) or \
                            (s == '1B' and '1B' in pos) or \
                            (s == '2B' and '2B' in pos) or \
                            (s == '3B' and '3B' in pos) or \
                            (s == 'SS' and 'SS' in pos) or \
                            (s.startswith('OF') and any(pos_tag in pos for pos_tag in ['OF', 'LF', 'CF', 'RF']))
                    if not valid: prob += x[p][s] == 0

            # Apply Stack Constraint
            team_hitters = eligible[
                (eligible['Team'] == current_team) & (~eligible['POS'].str.contains('P'))].index.tolist()
            if len(team_hitters) >= min_stack:
                prob += pulp.lpSum([x[p][s] for p in team_hitters for s in SLOTS]) >= min_stack

            # Diversity Logic
            for past in used_player_indices:
                prob += pulp.lpSum([x[p][s] for p in past for s in SLOTS]) <= (len(SLOTS) - diversity)

            prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=2))

            if pulp.LpStatus[prob.status] == 'Optimal':
                score = pulp.value(prob.objective)
                if score > highest_score:
                    highest_score = score
                    l_players, p_indices, t_sal, t_proj = [], [], 0, 0
                    for p in indices:
                        for s in SLOTS:
                            if pulp.value(x[p][s]) == 1:
                                r = eligible.loc[p]
                                l_players.append({
                                    'Slot': s, 'Name': r['Player'], 'Salary': int(r['Salary']),
                                    'Team': r['Team'], 'Opponent': r['Opponent'],
                                    'Logo': get_logo_url(r['Team']),
                                    'SortKey': POS_ORDER[s],
                                    'Order': r.get('Order', 0),
                                    'Hand': r.get('CleanHand', '')  # or whatever your hand column is named
                                })
                                p_indices.append(p)
                                t_sal += r['Salary']
                                t_proj += r['Proj_Base']
                    l_players.sort(key=lambda x: x['SortKey'])
                    best_lineup = {'players': l_players, 'total_salary': int(t_sal),
                                   'total_projection': round(t_proj, 2), 'indices': p_indices}

        if best_lineup:
            all_results.append(best_lineup)
            used_player_indices.append(best_lineup['indices'])
        else:
            break
    return all_results


@mlb_bp.route('/', methods=['GET', 'POST'])
def index():
    df_raw = get_df_raw()
    if df_raw.empty: return "MLB Offline", 503

    # Define SEO Matchup early to avoid NameError
    dynamic_matchup = get_prime_matchup()

    # Fetch fresh weather and game data
    weather = get_mlb_weather_data()
    espn = get_espn_game_info()
    h_fg, p_fg = get_weighted_stats()

    pool_ui, game_map = [], {}
    choices_h = [str(n) for n in h_fg['full_name'].dropna().tolist()]
    choices_p = [str(n) for n in p_fg['full_name'].dropna().tolist()]

    for idx, row in df_raw.iterrows():
        name = str(row.get('Player', ''))
        is_p = 'P' in str(row['POS'])

        # 1. Attempt ID and Stat Match
        mlb_id = "0"
        adv = {}
        try:
            choices = choices_p if is_p else choices_h
            stats_df = p_fg if is_p else h_fg
            # Threshold set to 80 to catch slight naming variations
            match = process.extractOne(name, choices, scorer=fuzz.token_set_ratio)
            if match and match[1] >= 80:
                adv = stats_df[stats_df['full_name'] == match[0]].iloc[0].to_dict()
                mlb_id = str(adv.get('mlb_id', '0'))
        except:
            pass

        # 2. Weather & Matchup (Standardization Fix)
        # Use TEAM_MAP to turn full names or alternate abbreviations into standardized keys
        t1 = TEAM_MAP.get(str(row['Team']), str(row['Team']))
        t2 = TEAM_MAP.get(str(row['Opponent']), str(row['Opponent']))
        g_id = " vs ".join(sorted([t1, t2]))

        w_data = weather.get(g_id, {})
        e_data = espn.get(g_id, {})

        # Build Matchup String safely using weather/espn fetched data
        time_val = e_data.get('time_str', 'TBD')
        venue_val = w_data.get('venue', 'Ballpark')
        temp_val = w_data.get('temp', '--')
        cond_val = w_data.get('condition', '')
        matchup_str = f"{time_val} @ {venue_val} | {temp_val}° {cond_val}"

        pool_ui.append({
            'Player': name,
            'POS': row['POS'],
            'Team': row['Team'],
            'Opponent': row['Opponent'],
            'Salary': int(row['Salary']),
            'Proj': round(row['Proj_Base'], 1),
            'Logo': get_logo_url(row['Team']),
            'Player_Image': get_player_headshot_url(mlb_id), # Reliable headshot/silhouette logic
            'Match_Display': matchup_str,
            'Primary_Stat': f"WHIP: {adv.get('WHIP', 1.35):.2f}" if is_p else f"ISO: {adv.get('ISO', 0.150):.3f}",
            'Secondary_Stat': f"K/9: {adv.get('K/9', 7.5):.1f}" if is_p else f"wRC+: {int(adv.get('wRC+', 100))}"
        })

        if g_id not in game_map:
            game_map[g_id] = {
                'id': g_id,
                'display': g_id,
                'time': time_val,
                'sort': e_data.get('raw_time', datetime.max.replace(tzinfo=pytz.utc))
            }

    # 3. Run Optimization if requested
    results, status = None, "MLB SYSTEMS ONLINE"
    if request.method == 'POST':
        results = run_optimizer(
            df_raw,
            num_lineups=int(request.form.get('num_lineups', 5)),
            locks=request.form.getlist('player_locks'),
            stack_team=request.form.get('stack_team'),
            min_stack=int(request.form.get('min_stack', 5)),
            diversity=int(request.form.get('diversity', 4)),
            excluded_games=request.form.getlist('excluded_games')
        )
        status = f"SUCCESS: {len(results)} LINEUPS" if results else "FAILED - CHECK CONSTRAINTS"

    return render_template('sport_mlb.html', sport="MLB", results=results, pool=pool_ui,
                           games=sorted(game_map.values(), key=lambda x: x['sort']),
                           matchup=dynamic_matchup, status=status, teams=sorted(df_raw['Team'].unique()))