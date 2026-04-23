import time
import random
import pandas as pd
import pulp
import traceback
import re
from flask import Blueprint, render_template, request
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
    # ... (Keep your existing get_clean_data logic as it is) ...
    try:
        sync_url = f"{NBA_SALARY_CSV}&t={int(time.time())}"
        df_raw = pd.read_csv(sync_url)
        if df_raw.empty: return pd.DataFrame()
        df_raw.columns = [str(c).strip() for c in df_raw.columns]
        clean_rows = []
        for _, row in df_raw.iterrows():
            try:
                name = str(row.get('Name', row.iloc[0])).split('·')[0].split('$')[0].strip()
                name = re.sub(r'\s[A-Z]{1,2}\d*$', '', name).strip()
                salary = 0
                if 'Salary' in row and pd.notnull(row['Salary']):
                    sal_val = str(row['Salary']).replace('$', '').replace(',', '')
                    salary = float(sal_val)
                elif 'Abbrev Salary' in row:
                    sal_val = str(row['Abbrev Salary']).replace('$', '').replace(',', '')
                    salary = float(sal_val)
                if salary > 0 and salary < 1000: salary *= 1000
                proj = 0
                if 'Projected Points' in row:
                    proj = float(row['Projected Points'])
                elif 'Proj' in row:
                    proj = float(row['Proj'])
                pos = str(row.get('POS', row.iloc[1])).upper().strip()
                team = str(row.get('Team', row.iloc[5])).upper().strip()
                opp = str(row.get('Opponent', row.iloc[6])).upper().strip()
                if name and name != 'nan' and salary > 0:
                    clean_rows.append(
                        {'Player': name, 'POS': pos, 'Salary': salary, 'Proj': proj, 'Team': team, 'Opponent': opp})
            except:
                continue
        df = pd.DataFrame(clean_rows)
        return df
    except Exception as e:
        print(f"!!! CRITICAL ERROR: {e}")
        return pd.DataFrame()


def run_nba_optimizer(df, num_lineups=1, diversity=3, exposure_limit=0.4, locks=[], excluded_games=[],
                      exposure_overrides={}):
    all_results, used_indices = [], []

    # 1. Game Filtering
    if excluded_games:
        df = df[~df.apply(lambda r: " vs ".join(sorted([str(r['Team']).strip().upper(),
                                                        str(r['Opponent']).strip().upper()])) in excluded_games,
                          axis=1)]

    # 2. Sequential Usage Tracker
    # We use player names to track across lineups
    player_usage = {name: 0 for name in df['Player'].unique()}

    for i in range(num_lineups):
        prob = pulp.LpProblem(f"NBA_Opt_{i}", pulp.LpMaximize)
        players = df.index.tolist()
        x = pulp.LpVariable.dicts("x", (players, NBA_SLOTS), cat="Binary")

        # 3. Objective with Exposure Check
        def get_mod_proj(row):
            # Check individual override first, then global limit
            limit_pct = exposure_overrides.get(row['Player'], exposure_limit)
            limit_count = max(1, int(num_lineups * limit_pct))

            # If hit limit, apply massive penalty to projection
            if player_usage.get(row['Player'], 0) >= limit_count:
                return -1000.0
            return row['Proj'] * random.uniform(0.98, 1.02)

        modified_projs = df.apply(get_mod_proj, axis=1)
        prob += pulp.lpSum([modified_projs[p] * x[p][s] for p in players for s in NBA_SLOTS])

        # 4. Constraints
        prob += pulp.lpSum([df.loc[p, 'Salary'] * x[p][s] for p in players for s in NBA_SLOTS]) <= 50000

        for s in NBA_SLOTS:
            prob += pulp.lpSum([x[p][s] for p in players]) == 1

        for p in players:
            prob += pulp.lpSum([x[p][s] for s in NBA_SLOTS]) <= 1
            if df.loc[p, 'Player'] in locks:
                prob += pulp.lpSum([x[p][s] for s in NBA_SLOTS]) == 1

            # POS Logic (PG, SG, SF, PF, C, G, F, UTIL)
            pos = str(df.loc[p, 'POS']).upper()
            for s in NBA_SLOTS:
                if s == 'UTIL':
                    valid = True
                elif s == 'G':
                    valid = any(pos_part in pos for pos_part in ['PG', 'SG', 'G'])
                elif s == 'F':
                    valid = any(pos_part in pos for pos_part in ['SF', 'PF', 'F'])
                else:
                    valid = (s == pos or s in pos.split('/'))
                if not valid: prob += x[p][s] == 0

        # Diversity (Overlap)
        for past in used_indices:
            prob += pulp.lpSum([x[p][s] for p in past for s in NBA_SLOTS]) <= (len(NBA_SLOTS) - diversity)

        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=2))

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
                        # Increment usage
                        player_usage[row['Player']] += 1

            lineup.sort(key=lambda x: SLOT_ORDER.get(x['Slot'], 99))
            all_results.append({'players': lineup, 'total_projection': round(t_proj, 2), 'total_salary': int(t_sal)})
            used_indices.append(p_indices)
        else:
            break
    return all_results


@nba_bp.route('/', methods=['GET', 'POST'])
def index():
    df = get_clean_data()
    if df.empty:
        return render_template('sport_nba.html', results=None, pool=[], sport="NBA", games=[],
                               status="WAITING FOR SLATE")

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
        except:
            continue

    results, status = None, "NBA SYSTEMS ONLINE"

    if request.method == 'POST':
        # 1. Handle Games
        included_games = request.form.getlist('games')
        all_game_ids = [g['id'] for g in unique_games]
        excluded = [gid for gid in all_game_ids if gid not in included_games]

        # 2. Collect Individual Exposure Overrides
        exposure_overrides = {}
        global_limit = float(request.form.get('exposure_limit', 0.4))
        for player in df['Player'].unique():
            val = request.form.get(f'exposure_{player}')
            if val:
                # Use individual limit, capped at global limit if desired,
                # or just use individual if it exists
                exposure_overrides[player] = float(val) / 100.0

        # 3. Run Optimizer
        results = run_nba_optimizer(
            df,
            num_lineups=int(request.form.get('num_lineups', 10)),
            diversity=int(request.form.get('diversity', 3)),
            exposure_limit=global_limit,
            locks=request.form.getlist('player_locks'),
            excluded_games=excluded,
            exposure_overrides=exposure_overrides
        )
        status = f"OPTIMIZED {len(results)} LINEUPS" if results else "OPTIMIZATION FAILED"

    return render_template('sport_nba.html', results=results, pool=pool_list,
                           sport="NBA", games=unique_games, status=status)