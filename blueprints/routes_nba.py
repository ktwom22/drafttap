from flask import Blueprint, render_template, request, current_app
import pandas as pd
import pulp
import random
import re
from helpers.nba_helpers import (
    get_nba_logo_url,
    get_nba_matchup_info,
    get_dynamic_espn_ids,
    get_player_headshot_url
)

nba_bp = Blueprint('nba', __name__, url_prefix='/nba')

# --- CONFIGURATION ---
NBA_SALARY_CSV = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTF0d2pT0myrD7vjzsB2IrEzMa3o1lylX5_GYyas_5UISsgOud7WffGDxSVq6tJhS45UaxFOX_FolyT/pub?gid=2055904356&single=true&output=csv"
NBA_SLOTS = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
SLOT_ORDER = {'PG': 0, 'SG': 1, 'SF': 2, 'PF': 3, 'C': 4, 'G': 5, 'F': 6, 'UTIL': 7}


def run_nba_optimizer(df, num_lineups=1, diversity=3, min_punts=1, exposure_limit=0.4, locks=[], excluded_games=[]):
    all_results, used_indices = [], []

    # Pre-filter projections and excluded games
    df = df[df['Proj'] > 0].copy()
    if excluded_games:
        df = df[
            ~df.apply(lambda r: " vs ".join(sorted([str(r['Team']), str(r['Opponent'])])) in excluded_games, axis=1)]

    player_usage = {p: 0 for p in df.index}
    max_use = max(1, int(num_lineups * exposure_limit))

    for i in range(num_lineups):
        prob = pulp.LpProblem(f"NBA_Opt_{i}", pulp.LpMaximize)
        players = df.index.tolist()
        x = pulp.LpVariable.dicts("x", (players, NBA_SLOTS), cat="Binary")

        # Objective
        prob += pulp.lpSum(
            [(df.loc[p, 'Proj'] * random.uniform(0.99, 1.01)) * x[p][s] for p in players for s in NBA_SLOTS])

        # Salary Cap
        prob += pulp.lpSum([df.loc[p, 'Salary'] * x[p][s] for p in players for s in NBA_SLOTS]) <= 50000

        # Punts
        punt_players = [p for p in players if df.loc[p, 'Salary'] < 4000]
        if punt_players and min_punts > 0:
            prob += pulp.lpSum([x[p][s] for p in punt_players for s in NBA_SLOTS]) >= min_punts

        # Fill all slots
        for s in NBA_SLOTS:
            prob += pulp.lpSum([x[p][s] for p in players]) == 1

        for p in players:
            prob += pulp.lpSum([x[p][s] for s in NBA_SLOTS]) <= 1
            if df.loc[p, 'Player'] in locks:
                prob += pulp.lpSum([x[p][s] for s in NBA_SLOTS]) == 1
            elif player_usage.get(p, 0) >= max_use:
                prob += pulp.lpSum([x[p][s] for s in NBA_SLOTS]) == 0

            # Position Eligibility
            pos = str(df.loc[p, 'POS'])
            for s in NBA_SLOTS:
                valid = (s == 'UTIL') or \
                        (s == 'G' and any(g in pos for g in ['PG', 'SG'])) or \
                        (s == 'F' and any(f in pos for f in ['SF', 'PF'])) or \
                        (s in pos)
                if not valid: prob += x[p][s] == 0

        for past in used_indices:
            prob += pulp.lpSum([x[p][s] for p in past for s in NBA_SLOTS]) <= (len(NBA_SLOTS) - diversity)

        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=5))

        if pulp.LpStatus[prob.status] == 'Optimal':
            lineup_players, p_indices, t_sal, t_proj = [], [], 0, 0
            for p in players:
                for s in NBA_SLOTS:
                    if pulp.value(x[p][s]) == 1:
                        row = df.loc[p]
                        lineup_players.append(
                            {'Slot': s, 'Name': row['Player'], 'Team': row['Team'], 'Salary': int(row['Salary'])})
                        p_indices.append(p)
                        t_sal += row['Salary']
                        t_proj += row['Proj']

            lineup_players.sort(key=lambda x: SLOT_ORDER.get(x['Slot'], 99))
            all_results.append(
                {'players': lineup_players, 'total_projection': round(t_proj, 2), 'total_salary': int(t_sal)})
            used_indices.append(p_indices)
            for idx in p_indices: player_usage[idx] += 1
        else:
            break
    return all_results


@nba_bp.route('/', methods=['GET', 'POST'])
def index():
    # Helper to get data (this is what we cache)
    def get_clean_data():
        try:
            raw_df = pd.read_csv(NBA_SALARY_CSV)
            raw_df.columns = raw_df.columns.str.strip()

            # Mapping
            name_col = next((c for c in ['Name', 'Player'] if c in raw_df.columns), 'Name')
            proj_col = next((c for c in ['Proj', 'FPTS', 'Projected Points'] if c in raw_df.columns), None)
            sal_col = next((c for c in ['Salary', 'salary'] if c in raw_df.columns), 'Salary')
            pos_col = next((c for c in ['POS', 'Pos'] if c in raw_df.columns), 'POS')
            team_col = next((c for c in ['Team', 'Abbrev'] if c in raw_df.columns), 'Team')
            opp_col = next((c for c in ['Opponent', 'Opp'] if c in raw_df.columns), 'Opponent')

            raw_df = raw_df.dropna(subset=[name_col])
            raw_df['Player'] = raw_df[name_col].astype(str).str.replace(' Q', '').str.strip()
            raw_df['Team'] = raw_df[team_col].astype(str).str.upper()
            raw_df['Opponent'] = raw_df[opp_col].astype(str).str.upper()
            raw_df['Salary'] = pd.to_numeric(raw_df[sal_col].astype(str).replace(r'[\$,kK]', '', regex=True),
                                             errors='coerce').fillna(0)
            if raw_df['Salary'].max() < 1000: raw_df['Salary'] *= 1000
            raw_df['Proj'] = pd.to_numeric(raw_df[proj_col], errors='coerce').fillna(0) if proj_col else 0.0
            raw_df['POS'] = raw_df[pos_col].fillna('UTIL')
            return raw_df
        except:
            return pd.DataFrame()

    df = get_clean_data()
    if df.empty:
        return "Error: Could not load data."

    # Build UI Pool
    dynamic_id_map = get_dynamic_espn_ids()
    pool_list, unique_games, seen_games = [], [], set()

    for _, r in df.iterrows():
        match_str = get_nba_matchup_info(r)
        espn_id = dynamic_id_map.get(r['Player'], "0")

        pool_list.append({
            'Player': r['Player'], 'POS': r['POS'], 'Team': r['Team'], 'Opponent': r['Opponent'],
            'Proj': round(r['Proj'], 1), 'Salary': int(r['Salary']),
            'Logo': get_nba_logo_url(r['Team']), 'Player_Image': get_player_headshot_url(espn_id),
            'Match_Display': match_str,
            'Primary_Stat': f"VAL: {round(r['Proj'] / (r['Salary'] / 1000), 1) if r['Salary'] > 0 else 0}x"
        })

        g_id = " vs ".join(sorted([str(r['Team']), str(r['Opponent'])]))
        if g_id not in seen_games:
            unique_games.append({'id': g_id, 'display': match_str})
            seen_games.add(g_id)

    results = None
    if request.method == 'POST':
        num = int(request.form.get('num_lineups', 10))
        div = int(request.form.get('diversity', 3))
        punts = int(request.form.get('min_punts', 1))
        exp = float(request.form.get('exposure_limit', 0.4))
        locks = request.form.getlist('player_locks')
        sel_games = request.form.getlist('games')

        all_ids = [g['id'] for g in unique_games]
        excluded = [gid for gid in all_ids if gid not in sel_games]

        results = run_nba_optimizer(df, num_lineups=num, diversity=div, min_punts=punts,
                                    exposure_limit=exp, locks=locks, excluded_games=excluded)

    return render_template(
        'sport_nba.html',
        results=results,
        pool=pool_list,
        sport="NBA",
        games=unique_games,
        status=f"OPTIMIZED {len(results)} LINEUPS" if results else "NBA SYSTEMS ONLINE"
    )