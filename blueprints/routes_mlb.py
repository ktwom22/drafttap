from flask import Blueprint, render_template, request
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
    TEAM_MAP,
    get_logo_url
)

mlb_bp = Blueprint('mlb', __name__, url_prefix='/mlb')

SALARY_CSV = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRzCRSTDnslz-zmGESH1CFhjsYD7NJa8yHkapMFu1JIR0M1PQDwZzMIDCmhPBUNU6kzLJy8-3_ioR4Y/pub?gid=1189680617&single=true&output=csv"

SLOTS = ['P1', 'P2', 'C', '1B', '2B', '3B', 'SS', 'OF1', 'OF2', 'OF3']
POS_ORDER = {s: i for i, s in enumerate(SLOTS)}


def clean_hand_str(h):
    if pd.isna(h): return "?"
    match = re.search(r'([RLS])', str(h).upper())
    return match.group(1) if match else "?"


def get_df_raw():
    try:
        sync_url = f"{SALARY_CSV}&t={int(time.time())}"
        df = pd.read_csv(sync_url)
        df.columns = df.columns.str.strip()

        # Salary & Projection Normalization
        for col, target in [('salary', 'Salary'), ('proj', 'Proj_Base')]:
            found = next((c for c in df.columns if col in c.lower()), None)
            df[target] = pd.to_numeric(df[found].astype(str).replace(r'[^0-9.]', '', regex=True),
                                       errors='coerce').fillna(0.0) if found else 0.0

        if 0 < df['Salary'].max() < 1000: df['Salary'] *= 1000
        df.loc[(df['Proj_Base'] <= 0.1) & (df['Salary'] > 0), 'Proj_Base'] = 5.0

        # Batting Order & Position
        order_col = next((c for c in df.columns if any(x in c.lower() for x in ['order', 'batting', 'bat  ✓'])), None)
        df['Order'] = df[order_col].astype(str).str.extract('(\d+)').fillna(0).astype(int) if order_col else 0
        df['POS'] = df['POS'].astype(str).str.upper().fillna('UTIL')

        # Handedness
        hand_col = next((c for c in df.columns if 'hand' in c.lower()), None)
        df['CleanHand'] = df[hand_col].apply(clean_hand_str) if hand_col else '?'

        return df
    except Exception as e:
        print(f"ERROR FETCHING CSV: {e}")
        return pd.DataFrame()


def run_optimizer(df_input, num_lineups=1, locks=[], stack_team=None, min_stack=5, diversity=4, excluded_games=[]):
    df = df_input.copy()
    if excluded_games:
        df = df[~df.apply(lambda r: " vs ".join(sorted([TEAM_MAP.get(str(r['Team']), str(r['Team'])),
                                                        TEAM_MAP.get(str(r['Opponent']),
                                                                     str(r['Opponent']))])) in excluded_games, axis=1)]

    all_results, used_player_indices = [], []
    p_hand_map = df[df['POS'].str.contains('P')].set_index('Team')['CleanHand'].to_dict()
    confirmed_teams = set(df[df['Order'] > 0]['Team'].unique())

    def apply_logic(row):
        proj = float(row['Proj_Base'])
        team, order = str(row['Team']), int(row.get('Order', 0))
        if 'P' in str(row['POS']):
            proj += (float(row.get('Chalk_Quality', 0)) * 0.5)
        else:
            if team in confirmed_teams:
                if order == 0: return 0.0
                proj *= (1.20 if order <= 2 else 1.10 if order <= 5 else 0.95)
            # Platoon Logic
            b_h, o_h = row['CleanHand'], p_hand_map.get(row['Opponent'], '?')
            if b_h == 'S' or (b_h == 'L' and o_h == 'R') or (b_h == 'R' and o_h == 'L'):
                proj *= 1.12
        return max(proj * random.uniform(0.97, 1.03), 0.1)

    df['Solver_Proj'] = df.apply(apply_logic, axis=1)
    eligible = df[df['Solver_Proj'] > 0.5].copy()
    if len(eligible) < 10: return []

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

                # Position Eligibility
                pos = str(eligible.loc[p, 'POS'])
                for s in SLOTS:
                    if (s.startswith('P') and 'P' not in pos) or \
                            (s == 'C' and 'C' not in pos) or \
                            (s == '1B' and '1B' not in pos) or \
                            (s == '2B' and '2B' not in pos) or \
                            (s == '3B' and '3B' not in pos) or \
                            (s == 'SS' and 'SS' not in pos) or \
                            (s.startswith('OF') and not any(x in pos for x in ['OF', 'LF', 'CF', 'RF'])):
                        prob += x[p][s] == 0

            # Stacking
            team_hitters = eligible[
                (eligible['Team'] == current_team) & (~eligible['POS'].str.contains('P'))].index.tolist()
            if len(team_hitters) >= min_stack:
                prob += pulp.lpSum([x[p][s] for p in team_hitters for s in SLOTS]) >= min_stack

            # Diversity
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
                                    'Slot': s, 'Name': r['Player'], 'Salary': r['Salary'],
                                    'Team': r['Team'], 'Opponent': r['Opponent'],
                                    'Logo': get_logo_url(r['Team']), 'Opp_Logo': get_logo_url(r['Opponent']),
                                    'Matchup_Hand': f"{r['CleanHand']} v {p_hand_map.get(r['Opponent'], '?')}",
                                    'SortKey': POS_ORDER[s]
                                })
                                p_indices.append(p);
                                t_sal += r['Salary'];
                                t_proj += r['Proj_Base']
                    l_players.sort(key=lambda x: x['SortKey'])
                    best_lineup = {'players': l_players, 'total_salary': t_sal, 'total_projection': round(t_proj, 2),
                                   'indices': p_indices}

        if best_lineup:
            all_results.append(best_lineup)
            used_player_indices.append(best_lineup['indices'])
        else:
            break
    return all_results


@mlb_bp.route('/', methods=['GET', 'POST'])
def index():
    weather = get_mlb_weather_data()  # Ensure this returns 'wind_speed' and 'wind_direction'
    espn = get_espn_game_info()
    h_fg, p_fg = get_weighted_stats()
    df_raw = get_df_raw()
    if df_raw.empty: return "Data Unavailable", 500

    pool_ui, game_map = [], {}
    for idx, row in df_raw.iterrows():
        is_p = 'P' in str(row['POS'])
        stats_df = p_fg if is_p else h_fg

        # Fuzzy Match Advanced Stats
        match = process.extractOne(row['Player'], stats_df['full_name'].tolist(), scorer=fuzz.token_set_ratio)
        adv = stats_df[stats_df['full_name'] == match[0]].iloc[0] if match and match[1] >= 80 else {}

        t1, t2 = TEAM_MAP.get(str(row['Team']), str(row['Team'])), TEAM_MAP.get(str(row['Opponent']),
                                                                                str(row['Opponent']))
        g_id = " vs ".join(sorted([t1, t2]))
        w = weather.get(g_id, {})

        p_data = row.to_dict()
        p_data.update({
            'Match_Display': g_id,
            'Weather_Short': f"{w.get('temp', '--')}°",
            'W_Icon': "🏟️" if "dome" in str(w.get('condition', '')).lower() else "☀️",
            'Wind_Speed': w.get('wind_speed', '0'),  # CRITICAL FOR UI
            'Wind_Direction': w.get('wind_direction', ''),  # CRITICAL FOR UI
            'Logo': get_logo_url(row['Team']),
            'Proj': round(row['Proj_Base'], 1),
            'Primary_Stat': f"WHIP: {adv.get('WHIP', 0):.2f}" if is_p else f"ISO: {adv.get('ISO', 0):.3f}",
            'Secondary_Stat': f"K/9: {adv.get('K/9', 0):.1f}" if is_p else f"wRC+: {int(adv.get('wRC+', 0))}"
        })
        pool_ui.append(p_data)

        if g_id not in game_map:
            info = espn.get(g_id, {'time_str': 'TBD', 'raw_time': datetime.max.replace(tzinfo=pytz.utc)})
            game_map[g_id] = {'id': g_id, 'display': g_id, 'time': info['time_str'], 'sort': info['raw_time']}

    results, status = None, "READY"
    if request.method == 'POST':
        results = run_optimizer(
            df_raw,
            num_lineups=int(request.form.get('num_lineups', 5)),
            locks=request.form.getlist('player_locks'),
            stack_team=request.form.get('stack_team'),
            min_stack=int(request.form.get('min_stack', 5)),
            diversity=int(request.form.get('diversity', 4))
        )
        status = f"SUCCESS: {len(results)} LINEUPS" if results else "FAILED"

    return render_template('sport_mlb.html', sport="MLB", results=results, status=status,
                           teams=sorted(df_raw['Team'].unique()),
                           games=sorted(game_map.values(), key=lambda x: x['sort']), pool=pool_ui)