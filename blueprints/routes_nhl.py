from flask import Blueprint, render_template, request
import pandas as pd
import pulp
import random
import re
from helpers.nhl_helpers import (
    get_nhl_logo_url,
    get_nhl_matchup_info,
    get_dynamic_nhl_ids,
    get_nhl_headshot_url
)

nhl_bp = Blueprint('nhl', __name__, url_prefix='/nhl')

# --- CONFIGURATION ---
NHL_SALARY_CSV = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRcEmUjeqWwtnYJFyT1T8CFRWR-sd-NEPZv4rZZ-BK8Rx3CYtWhDHb9ZbNMhkiaExaPHeqt-eoPNH2-/pub?gid=1627698462&single=true&output=csv"

NHL_SLOTS = ['C1', 'C2', 'W1', 'W2', 'W3', 'D1', 'D2', 'G', 'UTIL']
SLOT_ORDER = {'C1': 0, 'C2': 1, 'W1': 2, 'W2': 3, 'W3': 4, 'D1': 5, 'D2': 6, 'G': 7, 'UTIL': 8}


def run_nhl_optimizer(df, num_lineups=1, diversity=3, min_punts=1, exposure_limit=0.4, locks=[], excluded_games=[],
                      stack_config=None):
    all_results, used_indices = [], []

    # Filter by games
    if excluded_games:
        df = df[
            ~df.apply(lambda r: " vs ".join(sorted([str(r['Team']), str(r['Opponent'])])) in excluded_games, axis=1)]

    print(f"--- OPTIMIZER DEBUG ---")
    print(f"Total Players in Pool: {len(df)}")

    player_usage = {p: 0 for p in df.index}
    max_use = max(1, int(num_lineups * exposure_limit))

    for i in range(num_lineups):
        prob = pulp.LpProblem(f"NHL_Opt_{i}", pulp.LpMaximize)
        players = df.index.tolist()
        x = pulp.LpVariable.dicts("x", (players, NHL_SLOTS), cat="Binary")

        # Objective (with slight randomness for variety)
        prob += pulp.lpSum(
            [(df.loc[p, 'Proj'] * random.uniform(0.97, 1.03)) * x[p][s] for p in players for s in NHL_SLOTS])

        # Salary Cap
        prob += pulp.lpSum([df.loc[p, 'Salary'] * x[p][s] for p in players for s in NHL_SLOTS]) <= 50000

        # Punts Logic
        punt_players = [p for p in players if df.loc[p, 'Salary'] < 3500]
        if punt_players and min_punts > 0:
            prob += pulp.lpSum([x[p][s] for p in punt_players for s in NHL_SLOTS]) >= min(min_punts, len(punt_players))

        # Basic Slot Constraints
        for s in NHL_SLOTS:
            prob += pulp.lpSum([x[p][s] for p in players]) == 1

        for p in players:
            prob += pulp.lpSum([x[p][s] for s in NHL_SLOTS]) <= 1
            if df.loc[p, 'Player'] in locks:
                prob += pulp.lpSum([x[p][s] for s in NHL_SLOTS]) == 1
            if player_usage.get(p, 0) >= max_use:
                prob += pulp.lpSum([x[p][s] for s in NHL_SLOTS]) == 0

            # Positional Logic
            pos = str(df.loc[p, 'POS']).upper()
            for s in NHL_SLOTS:
                if s == 'G':
                    if 'G' not in pos: prob += x[p][s] == 0
                elif s == 'UTIL':
                    if 'G' in pos: prob += x[p][s] == 0
                else:
                    if s[0] not in pos: prob += x[p][s] == 0

        # --- STACKING LOGIC ---
        if stack_config and "-" in stack_config:
            sizes = [int(s) for s in stack_config.split('-')]
            teams = df['Team'].unique()
            t_stack = pulp.LpVariable.dicts("t_stack", (teams, sizes), cat="Binary")

            for size in sizes:
                prob += pulp.lpSum([t_stack[t][size] for t in teams]) == 1
                for t in teams:
                    t_idx = df[df['Team'] == t].index.tolist()
                    prob += pulp.lpSum([x[p][s] for p in t_idx for s in NHL_SLOTS if s != 'G']) >= size * t_stack[t][
                        size]

        # --- GOALIE ANTI-CORRELATION ---
        for p in players:
            if 'G' in str(df.loc[p, 'POS']):
                opp = df.loc[p, 'Opponent']
                opp_players = df[df['Team'] == opp].index.tolist()
                for p_opp in opp_players:
                    prob += pulp.lpSum([x[p]['G']]) + pulp.lpSum([x[p_opp][s] for s in NHL_SLOTS if s != 'G']) <= 1

        # Diversity
        for past in used_indices:
            prob += pulp.lpSum([x[p][s] for p in past for s in NHL_SLOTS]) <= (len(NHL_SLOTS) - diversity)

        prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=5))

        if pulp.LpStatus[prob.status] == 'Optimal':
            lineup, p_indices, t_sal, t_proj = [], [], 0, 0
            for p in players:
                for s in NHL_SLOTS:
                    if pulp.value(x[p][s]) == 1:
                        row = df.loc[p]
                        lineup.append({'Slot': s, 'Name': row['Player'], 'Team': row['Team'], 'Salary': row['Salary']})
                        p_indices.append(p)
                        t_sal += row['Salary']
                        t_proj += row['Proj']
            lineup.sort(key=lambda x: SLOT_ORDER.get(x['Slot'], 99))
            all_results.append({'players': lineup, 'total_projection': round(t_proj, 2), 'total_salary': t_sal})
            used_indices.append(p_indices)
            for idx in p_indices: player_usage[idx] += 1
        else:
            break
    return all_results


@nhl_bp.route('/', methods=['GET', 'POST'])
def index():
    try:
        df_raw = pd.read_csv(NHL_SALARY_CSV)
        clean_rows = []
        for _, row in df_raw.iterrows():
            try:
                raw_str = str(row.iloc[0])

                # 1. Extraction: Get the name part
                name_part = raw_str.split('·')[0].split('$')[0].strip()

                # 2. Cleanup: Remove trailing Position/Rank letters and digits (e.g., "W11", "D1")
                # This ensures the helper looks up "Cale Makar" instead of "Cale Makar D11"
                name = re.sub(r'\s[A-Z]\d+$', '', name_part).strip()

                # 3. Cleanup: Strip out common injury/status tags
                for tag in ["DTD", "IR", "O", "OUT"]:
                    if name.endswith(tag):
                        name = name[:-len(tag)].strip()

                salary_val = raw_str.split('$')[1].split(' ')[0].replace(',', '') if '$' in raw_str else "0"
                proj_val = raw_str.split('FPTS')[1].strip().split(' ')[0] if 'FPTS' in raw_str else "0"

                clean_rows.append({
                    'Player': name,
                    'POS': str(row.iloc[1]).upper().strip(),
                    'Salary': float(salary_val) * (1000 if float(salary_val) < 1000 else 1),
                    'Proj': float(proj_val),
                    'Team': str(row.iloc[4]).upper().strip(),
                    'Opponent': str(row.iloc[5]).upper().strip()
                })
            except:
                continue
        df = pd.DataFrame(clean_rows)
    except Exception as e:
        return f"Error loading data: {e}"

    dynamic_id_map = get_dynamic_nhl_ids()
    pool_list, unique_games, seen_games = [], [], set()

    for _, r in df.iterrows():
        match_str = get_nhl_matchup_info(r)
        pool_list.append({
            'Player': r['Player'], 'POS': r['POS'], 'Team': r['Team'], 'Opponent': r['Opponent'],
            'Proj': r['Proj'], 'Salary': int(r['Salary']),
            'Logo': get_nhl_logo_url(r['Team']),
            'Player_Image': get_nhl_headshot_url(r['Player'], dynamic_id_map),
            'Match_Display': match_str
        })
        g_id = " vs ".join(sorted([r['Team'], r['Opponent']]))
        if g_id not in seen_games:
            unique_games.append({'id': g_id, 'display': match_str})
            seen_games.add(g_id)

    results = None
    if request.method == 'POST':
        num = int(request.form.get('num_lineups', 10))
        div = int(request.form.get('diversity', 3))
        locks = request.form.getlist('player_locks')
        stack = request.form.get('stack_config', '')
        sel_games = request.form.getlist('games')

        all_ids = [g['id'] for g in unique_games]
        excluded = [gid for gid in all_ids if gid not in sel_games]

        results = run_nhl_optimizer(df, num_lineups=num, diversity=div, locks=locks,
                                    excluded_games=excluded, stack_config=stack)

    return render_template('sport_nhl.html', results=results, pool=pool_list, sport="NHL", games=unique_games)