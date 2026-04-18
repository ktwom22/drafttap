from flask import Blueprint, render_template, request
import pandas as pd
import pulp
import random
from helpers.nba_helpers import get_nba_logo_url

nba_bp = Blueprint('nba', __name__, url_prefix='/nba')

NBA_SALARY_CSV = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTF0d2pT0myrD7vjzsB2IrEzMa3o1lylX5_GYyas_5UISsgOud7WffGDxSVq6tJhS45UaxFOX_FolyT/pub?gid=2055904356&single=true&output=csv"

# DraftKings NBA Roster Slots
NBA_SLOTS = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
SLOT_ORDER = {
    'PG': 0, 'SG': 1, 'SF': 2, 'PF': 3,
    'C': 4, 'G': 5, 'F': 6, 'UTIL': 7
}


def run_nba_optimizer(df, num_lineups=1, diversity=3, min_punts=1, exposure_limit=0.4, locks=[], excluded_games=[]):
    all_results = []
    used_indices = []

    # Filter out excluded games from the Sidebar
    if excluded_games:
        df = df[
            ~df.apply(lambda r: " vs ".join(sorted([str(r['Team']), str(r['Opponent'])])) in excluded_games, axis=1)]

    player_usage = {p: 0 for p in df.index}
    max_use = max(1, int(num_lineups * exposure_limit))

    for i in range(num_lineups):
        prob = pulp.LpProblem(f"NBA_Opt_{i}", pulp.LpMaximize)
        players = df.index.tolist()
        x = pulp.LpVariable.dicts("x", (players, NBA_SLOTS), cat="Binary")

        # Objective: Maximize Projection (with small randomness to prevent identical lineups)
        prob += pulp.lpSum(
            [(df.loc[p, 'Proj'] * random.uniform(0.99, 1.01)) * x[p][s] for p in players for s in NBA_SLOTS])

        # Punt play constraint (< $4k) from Command Center
        punt_players = [p for p in players if df.loc[p, 'Salary'] < 4000]
        if punt_players and min_punts > 0:
            prob += pulp.lpSum([x[p][s] for p in punt_players for s in NBA_SLOTS]) >= min_punts

        # Salary Cap
        prob += pulp.lpSum([df.loc[p, 'Salary'] * x[p][s] for p in players for s in NBA_SLOTS]) <= 50000

        # Position Eligibility
        for s in NBA_SLOTS:
            prob += pulp.lpSum([x[p][s] for p in players]) == 1

        for p in players:
            prob += pulp.lpSum([x[p][s] for s in NBA_SLOTS]) <= 1

            # Player Locks
            if df.loc[p, 'Player'] in locks:
                prob += pulp.lpSum([x[p][s] for s in NBA_SLOTS]) == 1

            # Global Exposure Limit
            if player_usage.get(p, 0) >= max_use:
                prob += pulp.lpSum([x[p][s] for s in NBA_SLOTS]) == 0

            pos = str(df.loc[p, 'POS'])
            for s in NBA_SLOTS:
                valid = False
                if s == 'UTIL':
                    valid = True
                elif s == 'G':
                    valid = any(g in pos for g in ['PG', 'SG'])
                elif s == 'F':
                    valid = any(f in pos for f in ['SF', 'PF'])
                elif s in pos:
                    valid = True
                if not valid: prob += x[p][s] == 0

        # Diversity Constraint
        for past in used_indices:
            prob += pulp.lpSum([x[p][s] for p in past for s in NBA_SLOTS]) <= (8 - diversity)

        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=5))

        if pulp.LpStatus[prob.status] == 'Optimal':
            lineup_players, p_indices, t_sal, t_proj = [], [], 0, 0
            for p in players:
                for s in NBA_SLOTS:
                    if pulp.value(x[p][s]) == 1:
                        row = df.loc[p]
                        lineup_players.append({
                            'Slot': s, 'Name': row['Player'], 'Team': row['Team'],
                            'Salary': row['Salary']
                        })
                        p_indices.append(p)
                        t_sal += row['Salary']
                        t_proj += row['Proj']

            lineup_players.sort(key=lambda x: SLOT_ORDER.get(x['Slot'], 99))
            all_results.append({
                'players': lineup_players,
                'total_projection': round(t_proj, 2),
                'total_salary': t_sal,
                'indices': p_indices
            })
            used_indices.append(p_indices)
            for idx in p_indices: player_usage[idx] += 1
        else:
            break
    return all_results


@nba_bp.route('/', methods=['GET', 'POST'])
def index():
    try:
        df = pd.read_csv(NBA_SALARY_CSV)
    except Exception as e:
        return f"Error loading CSV: {e}"

    df.columns = df.columns.str.strip()

    # Column Mapping
    name_col = next((c for c in ['Name', 'Player', 'Alt Name'] if c in df.columns), 'Name')
    proj_col = next((c for c in ['Projected Points', 'Proj', 'FPTS'] if c in df.columns), None)
    sal_col = next((c for c in ['Salary', 'salary', 'Sal'] if c in df.columns), 'Salary')
    pos_col = next((c for c in ['POS', 'Pos'] if c in df.columns), 'POS')
    team_col = next((c for c in ['Team', 'Abbrev'] if c in df.columns), 'Team')
    opp_col = next((c for c in ['Opponent', 'Opp'] if c in df.columns), 'Opponent')
    start_col = next((c for c in ['Start', 'Time'] if c in df.columns), 'Start')

    # Data Cleaning
    df = df.dropna(subset=[name_col])
    df['Player'] = df[name_col].astype(str).str.replace('&nbsp;Q', '').str.replace(' Q', '').str.strip()
    df['Team'] = df[team_col].astype(str)
    df['Opponent'] = df[opp_col].astype(str)
    df['Start_Time'] = df[start_col].astype(str).replace('nan', 'TBD')
    df['Salary'] = pd.to_numeric(df[sal_col].astype(str).replace(r'[\$,kK]', '', regex=True), errors='coerce').fillna(0)
    if df['Salary'].max() < 1000: df['Salary'] *= 1000
    df['Proj'] = pd.to_numeric(df[proj_col], errors='coerce').fillna(0) if proj_col else 0.0
    df['POS'] = df[pos_col].fillna('UTIL')

    # Build Player Pool for Sidebar
    pool_list = []
    for _, r in df.iterrows():
        # Calculate Value (PTS per $1k)
        val_score = round(r['Proj'] / (r['Salary'] / 1000), 2) if r['Salary'] > 0 else 0
        pool_list.append({
            'Player': r['Player'], 'POS': r['POS'], 'Team': r['Team'], 'Opponent': r['Opponent'],
            'Proj': round(r['Proj'], 1), 'Salary': int(r['Salary']),
            'Logo': get_nba_logo_url(r['Team']), 'Match_Display': f"{r['Team']} @ {r['Opponent']}",
            'Primary_Stat': f"VAL: {val_score}x", 'Secondary_Stat': f"TIP: {r['Start_Time']}"
        })

    # Build Games list for Command Center checkboxes
    unique_games = []
    seen_games = set()
    for _, r in df.iterrows():
        g_id = " vs ".join(sorted([str(r['Team']), str(r['Opponent'])]))
        if g_id not in seen_games:
            unique_games.append({
                'id': g_id, 'display': f"{r['Team']} @ {r['Opponent']}", 'time': r['Start_Time']
            })
            seen_games.add(g_id)

    results, status = None, "NBA SYSTEMS ONLINE"

    if request.method == 'POST':
        # Retrieve Command Center inputs
        num = int(request.form.get('num_lineups', 10))
        div = int(request.form.get('diversity', 3))
        punts = int(request.form.get('min_punts', 1))
        exp = float(request.form.get('exposure_limit', 0.4))
        locks = request.form.getlist('player_locks')
        sel_games = request.form.getlist('games')

        # Identify excluded games
        all_ids = [g['id'] for g in unique_games]
        excluded = [gid for gid in all_ids if gid not in sel_games]

        results = run_nba_optimizer(
            df, num_lineups=num, diversity=div,
            min_punts=punts, exposure_limit=exp,
            locks=locks, excluded_games=excluded
        )
        status = f"OPTIMIZED {len(results)} LINEUPS" if results else "INFEASIBLE SLATE"

    return render_template(
        'sport_nba.html',  # Specific NBA Template
        results=results,
        pool=pool_list,
        sport="NBA",
        games=unique_games,
        status=status
    )