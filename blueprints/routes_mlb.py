import time
import random
import pandas as pd
import pulp
import traceback
import re
from flask import Blueprint, render_template, request, current_app
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

# --- CONFIGURATION ---
SALARY_CSV = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRzCRSTDnslz-zmGESH1CFhjsYD7NJa8yHkapMFu1JIR0M1PQDwZzMIDCmhPBUNU6kzLJy8-3_ioR4Y/pub?gid=1189680617&single=true&output=csv"
SLOTS = ['P1', 'P2', 'C', '1B', '2B', '3B', 'SS', 'OF1', 'OF2', 'OF3']
POS_ORDER = {s: i for i, s in enumerate(SLOTS)}


def get_df_raw():
    try:
        sync_url = f"{SALARY_CSV}&t={int(time.time())}"
        df = pd.read_csv(sync_url)
        df.columns = df.columns.str.strip()

        # Salary & Projection Mapping
        for col, target in [('salary', 'Salary'), ('proj', 'Proj_Base')]:
            found = next((c for c in df.columns if col in c.lower()), None)
            if found:
                df[target] = pd.to_numeric(df[found].astype(str).replace(r'[^0-9.]', '', regex=True),
                                           errors='coerce').fillna(0.0)
            else:
                df[target] = 0.0

        if 0 < df['Salary'].max() < 1000: df['Salary'] *= 1000

        # Batting Order Extraction
        order_col = next((c for c in df.columns if any(x in c.lower() for x in ['order', 'batting', 'bat  ✓'])), None)
        if order_col:
            df['Order'] = df[order_col].astype(str).str.extract('(\d+)').fillna(0).astype(int)
        else:
            df['Order'] = 0

        df['POS'] = df['POS'].astype(str).str.upper().fillna('UTIL')
        df['Player'] = df['Player'].astype(str).str.strip()
        return df
    except Exception as e:
        print(f"MLB CSV ERROR: {e}")
        return pd.DataFrame()


def run_optimizer(df_input, num_lineups=1, locks=[], stack_team=None, min_stack=4,
                  diversity=4, excluded_games=[], exposure_limits={}):
    df = df_input.copy()

    # Sequential Usage Tracking
    player_usage = {name: 0 for name in df['Player'].unique()}

    # Filter Excluded Games
    if excluded_games:
        df = df[~df.apply(lambda r: " vs ".join(sorted([TEAM_MAP.get(str(r['Team']), str(r['Team'])),
                                                        TEAM_MAP.get(str(r['Opponent']),
                                                                     str(r['Opponent']))])) in excluded_games, axis=1)]

    all_results, used_player_indices = [], []

    # Identify Valid Teams for Stacking (Must have enough hitters in the pool)
    valid_teams = df[~df['POS'].str.contains('P')].groupby('Team').filter(lambda x: len(x) >= min_stack)[
        'Team'].unique().tolist()

    if stack_team and stack_team != "None":
        target_teams = [stack_team] if stack_team in valid_teams else []
    else:
        # Rank teams by average of top 6 projections
        target_teams = df[~df['POS'].str.contains('P')].groupby('Team')['Proj_Base'].nlargest(6).groupby(
            'Team').mean().sort_values(ascending=False).index.tolist()
        target_teams = [t for t in target_teams if t in valid_teams]

    if not target_teams:
        return []

    for i in range(num_lineups):
        best_lineup, highest_score = None, -1

        if not stack_team or stack_team == "None":
            random.shuffle(target_teams)

        # Iterate through the top 15 most viable stacking options
        for current_team in target_teams[:15]:
            prob = pulp.LpProblem(f"MLB_{i}_{current_team}", pulp.LpMaximize)
            indices = df.index.tolist()
            x = pulp.LpVariable.dicts("x", (indices, SLOTS), cat="Binary")

            # Batting Order Multipliers & Exposure Safety Valve
            def get_modified_proj(row):
                # Hard cap via massive penalty
                if player_usage.get(row['Player'], 0) >= exposure_limits.get(row['Player'], num_lineups):
                    return -5000.0

                p = row['Proj_Base']
                if 'P' not in str(row['POS']):
                    o = row.get('Order', 0)
                    if 1 <= o <= 2:
                        p *= 1.25
                    elif 3 <= o <= 5:
                        p *= 1.15
                    elif o > 5:
                        p *= 0.90
                # Add Jitter for GPP diversity
                return p + random.uniform(-0.15, 0.15)

            modified_projs = df.apply(get_modified_proj, axis=1)
            prob += pulp.lpSum([modified_projs[p] * x[p][s] for p in indices for s in SLOTS])

            # Salary Cap
            prob += pulp.lpSum([df.loc[p, 'Salary'] * x[p][s] for p in indices for s in SLOTS]) <= 50000

            # Slot Constraints
            for s in SLOTS: prob += pulp.lpSum([x[p][s] for p in indices]) == 1
            for p in indices:
                prob += pulp.lpSum([x[p][s] for s in SLOTS]) <= 1
                if df.loc[p, 'Player'] in locks: prob += pulp.lpSum([x[p][s] for s in SLOTS]) == 1

                # Position Logic
                pos = str(df.loc[p, 'POS'])
                for s in SLOTS:
                    valid = (s.startswith('P') and 'P' in pos) or \
                            (s == 'C' and 'C' in pos) or \
                            ('1B' in s and '1B' in pos) or \
                            ('2B' in s and '2B' in pos) or \
                            ('3B' in s and '3B' in pos) or \
                            ('SS' in s and 'SS' in pos) or \
                            ('OF' in s and any(x in pos for x in ['OF', 'LF', 'CF', 'RF']))
                    if not valid: prob += x[p][s] == 0

            # --- STACKING ENGINE ---
            hitter_slots = [s for s in SLOTS if not s.startswith('P')]
            team_hitter_idx = df[(df['Team'] == current_team) & (~df['POS'].str.contains('P'))].index.tolist()
            prob += pulp.lpSum([x[p][s] for p in team_hitter_idx for s in hitter_slots]) == min_stack

            # Diversity (Overlap)
            for past in used_player_indices:
                prob += pulp.lpSum([x[p][s] for p in past for s in SLOTS]) <= (len(SLOTS) - diversity)

            prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=1.5))

            if pulp.LpStatus[prob.status] == 'Optimal':
                score = pulp.value(prob.objective)
                if score > highest_score:
                    highest_score = score
                    l_players, p_indices, t_sal, t_proj = [], [], 0, 0
                    for p in indices:
                        for s in SLOTS:
                            if pulp.value(x[p][s]) == 1:
                                r = df.loc[p]
                                l_players.append({
                                    'Slot': s, 'Name': r['Player'], 'Salary': int(r['Salary']),
                                    'Team': r['Team'], 'Opponent': r['Opponent'],
                                    'Logo': get_logo_url(r['Team']), 'SortKey': POS_ORDER[s],
                                    'Order': r.get('Order', 0)
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
            for pl in best_lineup['players']: player_usage[pl['Name']] += 1

            # Remove the team from the primary pool if we aren't forcing it, to ensure variety
            if not stack_team or stack_team == "None":
                used_team = best_lineup['players'][2]['Team']
                if used_team in target_teams: target_teams.remove(used_team)
        else:
            break

    return all_results


@mlb_bp.route('/', methods=['GET', 'POST'])
def index():
    df_raw = get_df_raw()
    if df_raw.empty: return "MLB Data Unreachable", 503

    dynamic_matchup = get_prime_matchup()
    weather = get_mlb_weather_data()
    espn = get_espn_game_info()
    h_fg, p_fg = get_weighted_stats()

    pool_ui, game_map = [], {}
    choices_h = [str(n) for n in h_fg['full_name'].dropna().tolist()]
    choices_p = [str(n) for n in p_fg['full_name'].dropna().tolist()]

    for idx, row in df_raw.iterrows():
        name = str(row.get('Player', ''))
        is_p = 'P' in str(row['POS'])
        mlb_id, adv = "0", {}
        try:
            choices = choices_p if is_p else choices_h
            stats_df = p_fg if is_p else h_fg
            match = process.extractOne(name, choices, scorer=fuzz.token_set_ratio)
            if match and match[1] >= 85:
                adv = stats_df[stats_df['full_name'] == match[0]].iloc[0].to_dict()
                mlb_id = str(adv.get('mlb_id', '0'))
        except:
            pass

        t1, t2 = TEAM_MAP.get(str(row['Team']), str(row['Team'])), TEAM_MAP.get(str(row['Opponent']),
                                                                                str(row['Opponent']))
        g_id = " vs ".join(sorted([t1, t2]))
        w_data, e_data = weather.get(g_id, {}), espn.get(g_id, {})
        time_val = e_data.get('time_str', 'TBD')
        matchup_str = f"{time_val} @ {w_data.get('venue', 'Ballpark')} | {w_data.get('temp', '--')}° {w_data.get('condition', '')}"

        pool_ui.append({
            'Player': name, 'POS': row['POS'], 'Team': row['Team'], 'Opponent': row['Opponent'],
            'Salary': int(row['Salary']), 'Proj': round(row['Proj_Base'], 1),
            'Logo': get_logo_url(row['Team']), 'Player_Image': get_player_headshot_url(mlb_id),
            'Match_Display': matchup_str,
            'Primary_Stat': f"WHIP: {adv.get('WHIP', 1.35):.2f}" if is_p else f"ISO: {adv.get('ISO', 0.150):.3f}",
            'Secondary_Stat': f"K/9: {adv.get('K/9', 7.5):.1f}" if is_p else f"wRC+: {int(adv.get('wRC+', 100))}"
        })

        if g_id not in game_map:
            game_map[g_id] = {'id': g_id, 'display': g_id, 'time': time_val,
                              'sort': e_data.get('raw_time', datetime.max.replace(tzinfo=pytz.utc))}

    results, status = None, "MLB SYSTEMS ONLINE"
    if request.method == 'POST':
        num_lineups = int(request.form.get('num_lineups', 5))
        global_cap = float(request.form.get('global_exposure_limit', 100))

        exposure_limits = {}
        for player_name in df_raw['Player'].unique():
            p_limit = float(request.form.get(f'exposure_{player_name}', 100))
            final_cap = min(p_limit, global_cap)
            exposure_limits[player_name] = max(1, int((final_cap / 100) * num_lineups))

        results = run_optimizer(
            df_raw,
            num_lineups=num_lineups,
            locks=request.form.getlist('player_locks'),
            stack_team=request.form.get('stack_team'),
            min_stack=int(request.form.get('min_stack', 4)),
            diversity=int(request.form.get('diversity', 4)),
            excluded_games=request.form.getlist('excluded_games'),
            exposure_limits=exposure_limits
        )
        status = f"OPTIMIZED {len(results)} LINEUPS" if results else "FAILED: CHECK CONSTRAINTS"

    return render_template('sport_mlb.html', sport="MLB", results=results, pool=pool_ui,
                           games=sorted(game_map.values(), key=lambda x: x['sort']),
                           matchup=dynamic_matchup, status=status, teams=sorted(df_raw['Team'].unique()))