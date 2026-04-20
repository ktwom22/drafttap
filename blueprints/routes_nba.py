import time
import random
import pandas as pd
import pulp
import traceback
from flask import Blueprint, render_template, request, current_app
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
SLOT_ORDER = {s: i for i, s in enumerate(NBA_SLOTS)}


def get_clean_data():
    """Fetches data and handles empty states or column mismatches gracefully."""
    try:
        sync_url = f"{NBA_SALARY_CSV}&t={int(time.time())}"
        raw_df = pd.read_csv(sync_url)

        if raw_df.empty:
            print("!!! NOTICE: Google Sheet is currently empty.")
            return pd.DataFrame()

        # Clean hidden whitespace from headers
        raw_df.columns = [str(c).strip() for c in raw_df.columns]

        # Explicit mapping based on your sheet's discovered headers
        name_col = next((c for c in raw_df.columns if c.lower() in ['name', 'player']), None)
        proj_col = next((c for c in raw_df.columns if c.lower() in ['projected points', 'proj', 'fpts']), None)
        sal_col = next((c for c in raw_df.columns if c.lower() in ['salary', 'sal']), None)
        pos_col = 'POS'
        team_col = 'Team'
        opp_col = 'Opponent'

        # Check for vital columns
        if not name_col or not sal_col:
            print(f"!!! WARNING: Vital columns missing. Found: {list(raw_df.columns)}")
            return pd.DataFrame()

        df = raw_df.dropna(subset=[name_col]).copy()

        # Data Sanitization
        df['Player'] = df[name_col].astype(str).str.replace(' Q', '').str.strip()
        df['Team'] = df[team_col].astype(str).str.upper().str.strip() if team_col in df.columns else "N/A"
        df['Opponent'] = df[opp_col].astype(str).str.upper().str.strip() if opp_col in df.columns else "N/A"

        # Salary Logic
        df['Salary'] = pd.to_numeric(df[sal_col].astype(str).replace(r'[\$,kK]', '', regex=True),
                                     errors='coerce').fillna(0)
        if df['Salary'].max() > 0 and df['Salary'].max() < 1000:
            df['Salary'] *= 1000

        # Projection Logic
        df['Proj'] = pd.to_numeric(df[proj_col], errors='coerce').fillna(0) if proj_col else 0.0
        df['POS'] = df[pos_col].astype(str).str.upper().str.strip() if pos_col in df.columns else "UTIL"

        # Final Filter: Only players with a name and a salary > 0
        df = df[(df['Player'] != 'nan') & (df['Salary'] > 0)]

        print(f"!!! SUCCESS: {len(df)} players loaded for the active slate.")
        return df

    except Exception as e:
        print(f"!!! DATA CLEANING ERROR: {e}")
        traceback.print_exc()
        return pd.DataFrame()


def run_nba_optimizer(df, num_lineups=1, diversity=3, min_punts=1, exposure_limit=0.4, locks=[], excluded_games=[]):
    all_results, used_indices = [], []
    # Filter for viable players
    df = df[df['Proj'] > 0].copy()

    if excluded_games:
        df = df[
            ~df.apply(lambda r: " vs ".join(sorted([str(r['Team']), str(r['Opponent'])])) in excluded_games, axis=1)]

    if df.empty: return []

    player_usage = {p: 0 for p in df.index}
    max_use = max(1, int(num_lineups * exposure_limit))

    for i in range(num_lineups):
        prob = pulp.LpProblem(f"NBA_Opt_{i}", pulp.LpMaximize)
        players = df.index.tolist()
        x = pulp.LpVariable.dicts("x", (players, NBA_SLOTS), cat="Binary")

        prob += pulp.lpSum(
            [(df.loc[p, 'Proj'] * random.uniform(0.99, 1.01)) * x[p][s] for p in players for s in NBA_SLOTS])
        prob += pulp.lpSum([df.loc[p, 'Salary'] * x[p][s] for p in players for s in NBA_SLOTS]) <= 50000

        punt_players = [p for p in players if df.loc[p, 'Salary'] < 4000]
        if punt_players and min_punts > 0:
            prob += pulp.lpSum([x[p][s] for p in punt_players for s in NBA_SLOTS]) >= min_punts

        for s in NBA_SLOTS:
            prob += pulp.lpSum([x[p][s] for p in players]) == 1

        for p in players:
            prob += pulp.lpSum([x[p][s] for s in NBA_SLOTS]) <= 1
            if df.loc[p, 'Player'] in locks:
                prob += pulp.lpSum([x[p][s] for s in NBA_SLOTS]) == 1
            elif player_usage.get(p, 0) >= max_use:
                prob += pulp.lpSum([x[p][s] for s in NBA_SLOTS]) == 0

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
            lineup, p_indices, t_sal, t_proj = [], [], 0, 0
            for p in players:
                for s in NBA_SLOTS:
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


@nba_bp.route('/', methods=['GET', 'POST'])
def index():
    df = get_clean_data()

    # Handle empty state: Pass empty lists to the template instead of a 500 error
    if df.empty:
        return render_template(
            'sport_nba.html',
            results=None,
            pool=[],
            sport="NBA",
            games=[],
            status="WAITING FOR SLATE"
        )

    dynamic_id_map = get_dynamic_espn_ids()
    pool_list, unique_games, seen_games = [], [], set()

    for _, r in df.iterrows():
        try:
            match_str = get_nba_matchup_info(r)
            pool_list.append({
                'Player': r['Player'], 'POS': r['POS'], 'Team': r['Team'], 'Opponent': r['Opponent'],
                'Proj': round(r['Proj'], 1), 'Salary': int(r['Salary']),
                'Logo': get_nba_logo_url(r['Team']),
                'Player_Image': get_player_headshot_url(dynamic_id_map.get(r['Player'], "0")),
                'Match_Display': match_str,
                'Value': round(r['Proj'] / (r['Salary'] / 1000), 1) if r['Salary'] > 0 else 0
            })

            g_id = " vs ".join(sorted([str(r['Team']), str(r['Opponent'])]))
            if g_id not in seen_games:
                unique_games.append({'id': g_id, 'display': match_str})
                seen_games.add(g_id)
        except Exception as e:
            continue

    results = None
    status = "NBA SYSTEMS ONLINE"

    if request.method == 'POST':
        results = run_nba_optimizer(
            df,
            num_lineups=int(request.form.get('num_lineups', 10)),
            diversity=int(request.form.get('diversity', 3)),
            min_punts=int(request.form.get('min_punts', 1)),
            exposure_limit=float(request.form.get('exposure_limit', 0.4)),
            locks=request.form.getlist('player_locks'),
            excluded_games=request.form.getlist('games')
        )
        status = f"OPTIMIZED {len(results)} LINEUPS" if results else "OPTIMIZATION FAILED"

    return render_template('sport_nba.html', results=results, pool=pool_list,
                           sport="NBA", games=unique_games, status=status)