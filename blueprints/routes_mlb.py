from flask import Blueprint, render_template, request
import pandas as pd
import pulp
import random
import re
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

# --- CONFIGURATION ---
SALARY_CSV = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRzCRSTDnslz-zmGESH1CFhjsYD7NJa8yHkapMFu1JIR0M1PQDwZzMIDCmhPBUNU6kzLJy8-3_ioR4Y/pub?gid=1189680617&single=true&output=csv"

POS_ORDER = {
    'P1': 0, 'P2': 1, 'C': 2, '1B': 3, '2B': 4,
    '3B': 5, 'SS': 6, 'OF1': 7, 'OF2': 8, 'OF3': 9
}


def clean_hand_str(h):
    if pd.isna(h): return "?"
    match = re.search(r'([RLS])', str(h).upper())
    return match.group(1) if match else "?"


def run_optimizer(df_input, num_lineups=1, locks=[], stack_team=None, min_stack=3, diversity=4, excluded_games=[],
                  exposure_limit=1.0, weather_data={}):
    df = df_input.copy()

    # 1. Filter Excluded Games
    if excluded_games:
        df = df[~df.apply(lambda r: " vs ".join(sorted([TEAM_MAP.get(str(r['Team']), str(r['Team'])),
                                                        TEAM_MAP.get(str(r['Opponent']),
                                                                     str(r['Opponent']))])) in excluded_games, axis=1)]

    all_results, used_player_indices = [], []
    player_usage = {p: 0 for p in df.index}
    max_count = max(1, int(num_lineups * exposure_limit))
    p_hand_map = df[df['POS'].str.contains('P', na=False)].set_index('Team')['CleanHand'].to_dict()

    def apply_logic(row):
        base_proj = float(row.get('Proj_Base', 5.0))
        proj = base_proj if base_proj > 0 else 5.0
        t1, t2 = TEAM_MAP.get(str(row['Team']), str(row['Team'])), TEAM_MAP.get(str(row['Opponent']),
                                                                                str(row['Opponent']))
        game_id = " vs ".join(sorted([t1, t2]))
        w = weather_data.get(game_id, {'temp': 70, 'wind': '0 mph, Calm', 'condition': 'Clear'})

        if 'P' in str(row['POS']):
            proj += (float(row.get('Chalk_Quality', 0)) * 0.8)
        else:
            order = row.get('Order', 0)
            if order == 0:
                proj *= 0.1
            elif order <= 2:
                proj *= 1.20
            elif order <= 5:
                proj *= 1.10

            # Handedness Advantage
            b_h, o_h = row.get('CleanHand', '?'), p_hand_map.get(row['Opponent'], '?')
            if b_h == 'S' or (b_h == 'L' and o_h == 'R') or (b_h == 'R' and o_h == 'L'):
                proj *= 1.15

            # Weather Adjustments
            wind, cond = str(w['wind']).lower(), str(w['condition']).lower()
            if "dome" not in cond and "closed" not in cond:
                try:
                    temp_val = int(re.search(r'\d+', str(w.get('temp', '70'))).group())
                    if temp_val >= 85: proj *= 1.07
                except:
                    pass
                if 'out' in wind:
                    proj *= 1.08
                elif 'in' in wind:
                    proj *= 0.92

            if float(row.get('Edge_Value', 0)) > 0:
                proj += (row['Edge_Value'] * 0.1)

        return proj * random.uniform(0.98, 1.02)

    df['Solver_Proj'] = df.apply(apply_logic, axis=1)

    # Stacking Logic Prep
    teams_to_stack = [stack_team] if stack_team and stack_team != "None" else \
        df[~df['POS'].str.contains('P')].groupby('Team')['Solver_Proj'].mean().sort_values(ascending=False).head(
            10).index.tolist()

    for i in range(num_lineups):
        best_lineup, highest_score = None, -1
        random.shuffle(teams_to_stack)

        for current_team in teams_to_stack[:4]:
            try:
                prob = pulp.LpProblem(f"MLB_{i}_{current_team}", pulp.LpMaximize)
                players, slots = df.index.tolist(), list(POS_ORDER.keys())
                x = pulp.LpVariable.dicts("x", (players, slots), cat="Binary")

                # Objective: Maximize Projections
                prob += pulp.lpSum([df.loc[p, 'Solver_Proj'] * x[p][s] for p in players for s in slots])

                # Constraints
                prob += pulp.lpSum([df.loc[p, 'Salary'] * x[p][s] for p in players for s in slots]) <= 50000
                for s in slots: prob += pulp.lpSum([x[p][s] for p in players]) == 1
                for p in players:
                    prob += pulp.lpSum([x[p][s] for s in slots]) <= 1
                    if df.loc[p, 'Player'] in locks: prob += pulp.lpSum([x[p][s] for s in slots]) == 1
                    if player_usage.get(p, 0) >= max_count: prob += pulp.lpSum([x[p][s] for s in slots]) == 0

                    # Position Logic
                    pos = str(df.loc[p, 'POS'])
                    for s in slots:
                        valid = any([(s.startswith('P') and 'P' in pos), (s == 'C' and 'C' in pos),
                                     (s == '1B' and '1B' in pos), (s == '2B' and '2B' in pos),
                                     (s == '3B' and '3B' in pos), (s == 'SS' and 'SS' in pos),
                                     (s.startswith('OF') and 'OF' in pos)])
                        if not valid: prob += x[p][s] == 0

                # Diversity Constraint
                for past in used_player_indices:
                    prob += pulp.lpSum([x[p][s] for p in past for s in slots]) <= (len(slots) - diversity)

                # Stack Constraint
                h_idx = df[(df['Team'] == current_team) & (~df['POS'].str.contains('P'))].index.tolist()
                if len(h_idx) >= int(min_stack) and int(min_stack) > 0:
                    prob += pulp.lpSum([x[p][s] for p in h_idx for s in slots]) >= int(min_stack)

                prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=4))

                if pulp.LpStatus[prob.status] == 'Optimal':
                    score = pulp.value(prob.objective)
                    if score > highest_score:
                        highest_score = score
                        lineup_data, p_indices, t_sal, t_proj = [], [], 0, 0
                        for p in players:
                            for s in slots:
                                if pulp.value(x[p][s]) == 1:
                                    row = df.loc[p]
                                    lineup_data.append({
                                        'Slot': s, 'Name': row['Player'], 'Team': row['Team'],
                                        'Opponent': row['Opponent'], 'Hand': row['CleanHand'],
                                        'Order': int(row['Order']), 'Proj': round(row['Proj_Base'], 2),
                                        'Salary': row['Salary'], 'SortKey': POS_ORDER[s]
                                    })
                                    p_indices.append(p);
                                    t_sal += row['Salary'];
                                    t_proj += row['Proj_Base']
                        lineup_data.sort(key=lambda x: x['SortKey'])
                        best_lineup = {'players': lineup_data, 'total_salary': t_sal,
                                       'total_projection': round(t_proj, 2), 'indices': p_indices}
            except:
                continue

        if best_lineup:
            all_results.append(best_lineup);
            used_player_indices.append(best_lineup['indices'])
            for idx in best_lineup['indices']: player_usage[idx] += 1
        else:
            break
    return all_results


@mlb_bp.route('/', methods=['GET', 'POST'])
def index():
    # 1. Fetch External Data
    weather_info = get_mlb_weather_data()
    game_data = get_espn_game_info()
    h_fg, p_fg = get_weighted_stats()

    # 2. Process CSV
    df_raw = pd.read_csv(SALARY_CSV)
    df_raw.columns = df_raw.columns.str.strip()

    # Map CSV Columns
    sal_col = next((c for c in ['Salary', 'salary', 'Sal'] if c in df_raw.columns), 'Salary')
    proj_col = next((c for c in ['Projected Points', 'Proj', 'FPTS'] if c in df_raw.columns), None)
    order_col = next((c for c in ['Batting Order', 'Order'] if c in df_raw.columns), None)
    hand_col = next((c for c in ['PlayerHand', 'Hand'] if c in df_raw.columns), None)

    df_raw['Salary'] = pd.to_numeric(df_raw[sal_col].astype(str).replace(r'[\$,]', '', regex=True),
                                     errors='coerce').fillna(0)
    df_raw['Proj_Base'] = pd.to_numeric(df_raw[proj_col], errors='coerce').fillna(0.0) if proj_col else 0.0
    df_raw['Order'] = pd.to_numeric(df_raw[order_col], errors='coerce').fillna(0).astype(int) if order_col else 0
    df_raw['CleanHand'] = df_raw[hand_col].apply(clean_hand_str) if hand_col else '?'

    # 3. Statistical Merging (Fuzzy Match)
    for col in ['Edge_Value', 'Chalk_Quality', 'ISO', 'wRC+', 'WHIP', 'HR/9', 'K/9', 'GS']: df_raw[col] = 0.0
    h_choices, p_choices = h_fg['full_name'].tolist(), p_fg['full_name'].tolist()

    for idx, row in df_raw.iterrows():
        is_pitcher = 'P' in str(row['POS'])
        choices = p_choices if is_pitcher else h_choices
        stats_df = p_fg if is_pitcher else h_fg
        m = process.extractOne(row['Player'], choices, scorer=fuzz.token_set_ratio)
        if m and m[1] >= 75:
            s_row = stats_df[stats_df['full_name'] == m[0]].iloc[0]
            cols = ['Chalk_Quality', 'WHIP', 'HR/9', 'K/9', 'GS'] if is_pitcher else ['Edge_Value', 'ISO', 'wRC+']
            for c in cols: df_raw.at[idx, c] = s_row[c]

    # 4. Build Player Pool & Game List
    pool_list, unique_games = [], {}
    for _, r in df_raw.iterrows():
        t1, t2 = TEAM_MAP.get(str(r['Team']), str(r['Team'])), TEAM_MAP.get(str(r['Opponent']), str(r['Opponent']))
        g_id = " vs ".join(sorted([t1, t2]))

        # Meta Info
        info = game_data.get(g_id)
        w = weather_info.get(g_id, {'temp': '--', 'wind': 'Calm', 'condition': 'Unknown'})
        matchup_str = f"vs {r['Opponent']}" if (info and r['Team'] == info['home_team']) else f"@ {r['Opponent']}"

        p_data = r.to_dict()
        p_data.update({
            'Match_Display': matchup_str, 'Weather_Short': f"{w['temp']}°", 'Logo': get_logo_url(r['Team']),
            'Proj': round(r['Proj_Base'], 1),
            'Primary_Stat': f"WHP: {r['WHIP']:.2f}" if 'P' in str(r['POS']) else f"ISO: {r['ISO']:.3f}",
            'Secondary_Stat': f"K/9: {r['K/9']:.1f}" if 'P' in str(r['POS']) else f"wRC: {int(r['wRC+'])}",
            'W_Icon': "🏟️" if ("dome" in str(w['condition']).lower() or "closed" in str(w['condition']).lower()) else (
                "🔥" if "out" in str(w['wind']).lower() else ("❄️" if "in" in str(w['wind']).lower() else "⚪"))
        })
        pool_list.append(p_data)

        if g_id not in unique_games:
            time_d = game_data.get(g_id,
                                   {'display': f"{t1} vs {t2}", 'raw_time': datetime.max.replace(tzinfo=pytz.utc)})
            unique_games[g_id] = {
                'id': g_id, 'display': time_d.get('display', g_id), 'time': time_d.get('time_str', 'TBD'),
                'sort': time_d.get('raw_time'), 'weather': f"{w['temp']}° {w['condition']}"
            }

    game_list = sorted(unique_games.values(), key=lambda x: x['sort'])

    # --- DYNAMIC SEO LOGIC ---
    sorted_pool = sorted(pool_list, key=lambda x: x['Proj'], reverse=True)
    top_p = next((p['Player'] for p in sorted_pool if 'P' in str(p['POS'])), "Elite Pitchers")
    top_h = next((p['Player'] for p in sorted_pool if 'P' not in str(p['POS'])), "Top Hitters")
    dynamic_stars = f"{top_p}, {top_h}"
    dynamic_matchups = ", ".join([g['display'] for g in game_list[:3]])

    results, status = None, "SYSTEMS LIVE"

    if request.method == 'POST':
        sel_games = request.form.getlist('games')
        excluded = [g['id'] for g in game_list if g['id'] not in sel_games]
        results = run_optimizer(
            df_raw,
            num_lineups=int(request.form.get('num_lineups', 10)),
            locks=request.form.getlist('player_locks'),
            stack_team=request.form.get('stack_team'),
            min_stack=int(request.form.get('min_stack', 3)),
            diversity=int(request.form.get('diversity', 4)),
            exposure_limit=float(request.form.get('exposure_limit', 1.0)),
            excluded_games=excluded,
            weather_data=weather_info
        )
        status = f"OPTIMIZED {len(results)} LINEUPS" if results else "SOLVER FAILED"

    return render_template(
        'sport_mlb.html',
        results=results,
        status=status,
        sport="MLB",
        teams=sorted(df_raw['Team'].dropna().unique()),
        games=game_list,
        pool=pool_list,
        top_stars=dynamic_stars,  # SEO Variable
        matchups=dynamic_matchups  # SEO Variable
    )