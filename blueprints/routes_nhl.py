from flask import Blueprint, render_template, request, current_app
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


def run_nhl_optimizer(df, num_lineups=1, diversity=3, exposure_limit=0.4, locks=[], excluded_games=[],
                      stack_config=None):
    all_results, used_indices = [], []

    # 1. Pre-filter and Exclude
    df = df[df['Proj'] > 0].copy()
    if excluded_games:
        df = df[
            ~df.apply(lambda r: " vs ".join(sorted([str(r['Team']), str(r['Opponent'])])) in excluded_games, axis=1)]

    player_usage = {p: 0 for p in df.index}
    max_use = max(1, int(num_lineups * exposure_limit))

    for i in range(num_lineups):
        prob = pulp.LpProblem(f"NHL_Opt_{i}", pulp.LpMaximize)
        players = df.index.tolist()
        x = pulp.LpVariable.dicts("x", (players, NHL_SLOTS), cat="Binary")

        # Objective with slight variance
        prob += pulp.lpSum(
            [(df.loc[p, 'Proj'] * random.uniform(0.97, 1.03)) * x[p][s] for p in players for s in NHL_SLOTS])

        # Salary Cap
        prob += pulp.lpSum([df.loc[p, 'Salary'] * x[p][s] for p in players for s in NHL_SLOTS]) <= 50000

        # Fill all slots
        for s in NHL_SLOTS:
            prob += pulp.lpSum([x[p][s] for p in players]) == 1

        for p in players:
            prob += pulp.lpSum([x[p][s] for s in NHL_SLOTS]) <= 1

            # Locks & Exposure
            if df.loc[p, 'Player'] in locks:
                prob += pulp.lpSum([x[p][s] for s in NHL_SLOTS]) == 1
            elif player_usage.get(p, 0) >= max_use:
                prob += pulp.lpSum([x[p][s] for s in NHL_SLOTS]) == 0

            # NHL Positional Logic
            pos = str(df.loc[p, 'POS']).upper()
            for s in NHL_SLOTS:
                if s == 'G':
                    if 'G' not in pos: prob += x[p][s] == 0
                elif s == 'UTIL':
                    if 'G' in pos: prob += x[p][s] == 0
                else:
                    # Matches 'C' to 'C1/C2', 'W' to 'W1/W2/W3', 'D' to 'D1/D2'
                    if s[0] not in pos: prob += x[p][s] == 0

        # Anti-Correlation: Don't play skaters against your starting Goalie
        for p in players:
            if 'G' in str(df.loc[p, 'POS']):
                opp = df.loc[p, 'Opponent']
                opp_skaters = df[(df['Team'] == opp) & (~df['POS'].str.contains('G'))].index.tolist()
                for skater_idx in opp_skaters:
                    prob += x[p]['G'] + pulp.lpSum([x[skater_idx][s] for s in NHL_SLOTS if s != 'G']) <= 1

        # Stacking Logic (e.g. "3-2" stack)
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

        # Diversity
        for past in used_indices:
            prob += pulp.lpSum([x[p][s] for p in past for s in NHL_SLOTS]) <= (len(NHL_SLOTS) - diversity)

        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=5))

        if pulp.LpStatus[prob.status] == 'Optimal':
            lineup, p_indices, t_sal, t_proj = [], [], 0, 0
            for p in players:
                for s in NHL_SLOTS:
                    if pulp.value(x[p][s]) == 1:
                        row = df.loc[p]
                        lineup.append(
                            {'Slot': s, 'Name': row['Player'], 'Team': row['Team'], 'Salary': int(row['Salary'])})
                        p_indices.append(p)
                        t_sal += row['Salary']
                        t_proj += row['Proj']
            lineup.sort(key=lambda x: SLOT_ORDER.get(x['Slot'], 99))
            all_results.append({'players': lineup, 'total_projection': round(t_proj, 2), 'total_salary': int(t_sal)})
            used_indices.append(p_indices)
            for idx in p_indices: player_usage[idx] += 1
        else:
            break
    return all_results


@nhl_bp.route('/', methods=['GET', 'POST'])
def index():
    def get_clean_nhl_data():
        try:
            df_raw = pd.read_csv(NHL_SALARY_CSV)
            clean_rows = []
            for _, row in df_raw.iterrows():
                try:
                    # Specific parsing for your NHL spreadsheet format
                    raw_str = str(row.iloc[0])
                    name_part = raw_str.split('·')[0].split('$')[0].strip()
                    name = re.sub(r'\s[A-Z]\d+$', '', name_part).strip()

                    # Clean injury tags
                    for tag in ["DTD", "IR", "O", "OUT"]:
                        if name.endswith(tag): name = name[:-len(tag)].strip()

                    sal_str = raw_str.split('$')[1].split(' ')[0].replace(',', '') if '$' in raw_str else "0"
                    proj_str = raw_str.split('FPTS')[1].strip().split(' ')[0] if 'FPTS' in raw_str else "0"

                    salary = float(sal_str)
                    if salary < 1000: salary *= 1000  # Handle 9.5k vs 9500

                    clean_rows.append({
                        'Player': name,
                        'POS': str(row.iloc[1]).upper().strip(),
                        'Salary': salary,
                        'Proj': float(proj_str),
                        'Team': str(row.iloc[4]).upper().strip(),
                        'Opponent': str(row.iloc[5]).upper().strip()
                    })
                except:
                    continue
            return pd.DataFrame(clean_rows)
        except:
            return pd.DataFrame()

    df = get_clean_nhl_data()
    if df.empty: return "Error loading NHL data."

    # UI Mapping
    dynamic_id_map = get_dynamic_nhl_ids()
    pool_list, unique_games, seen_games = [], [], set()

    for _, r in df.iterrows():
        match_str = get_nhl_matchup_info(r)
        pool_list.append({
            'Player': r['Player'], 'POS': r['POS'], 'Team': r['Team'],
            'Opponent': r['Opponent'], 'Proj': r['Proj'], 'Salary': int(r['Salary']),
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
        exp = float(request.form.get('exposure_limit', 0.4))
        locks = request.form.getlist('player_locks')
        stack = request.form.get('stack_config', '')
        sel_games = request.form.getlist('games')

        all_ids = [g['id'] for g in unique_games]
        excluded = [gid for gid in all_ids if gid not in sel_games]

        results = run_nhl_optimizer(df, num_lineups=num, diversity=div, exposure_limit=exp,
                                    locks=locks, excluded_games=excluded, stack_config=stack)

    return render_template('sport_nhl.html', sport="NHL", results=results, pool=pool_list,
                           games=unique_games, status=f"READY: {len(df)} PLAYERS")