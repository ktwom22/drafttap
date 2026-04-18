import pulp
import random
from helpers.mlb_helpers import TEAM_MAP, get_logo_url

POS_ORDER = {'P1': 0, 'P2': 1, 'C': 2, '1B': 3, '2B': 4, '3B': 5, 'SS': 6, 'OF1': 7, 'OF2': 8, 'OF3': 9}


def run_optimizer(df_input, num_lineups=1, locks=[], stack_team=None, min_stack=3, diversity=4,
                  excluded_games=[], exposure_limit=1.0, weather_data={}, game_statuses={}):
    """
    game_statuses: dict mapping GID ("TEAM vs OPP") to boolean (True if started/locked)
    """
    df = df_input.copy()

    # 1. Filter excluded games
    if excluded_games:
        df = df[~df.apply(lambda r: " vs ".join(sorted([TEAM_MAP.get(str(r['Team']), str(r['Team'])),
                                                        TEAM_MAP.get(str(r['Opponent']),
                                                                     str(r['Opponent']))])) in excluded_games, axis=1)]

    all_results, used_player_indices = [], []
    player_usage = {p: 0 for p in df.index}
    max_count = max(1, int(num_lineups * exposure_limit))
    p_hand_map = df[df['POS'].str.contains('P', na=False)].set_index('Team')['CleanHand'].to_dict()

    def apply_logic(row):
        t1, t2 = TEAM_MAP.get(str(row['Team']), str(row['Team'])), TEAM_MAP.get(str(row['Opponent']),
                                                                                str(row['Opponent']))
        gid = " vs ".join(sorted([t1, t2]))

        # Check lock status
        is_locked = game_statuses.get(gid, False)

        # Base projection from your sheet/source
        proj_val = row.get('Kris Bubic projected points', row.get('Proj', 5.0))
        proj = float(proj_val) if float(proj_val) > 0 else 5.0

        # IF LOCKED: Return raw projection immediately (No weather, No hand adjustments, No randomness)
        if is_locked:
            return proj

        # IF NOT LOCKED: Apply dynamic logic
        w = weather_data.get(gid, {'temp': 70, 'wind': '0 mph, Calm', 'condition': 'Clear'})

        if 'P' in str(row['POS']):
            proj += (float(row.get('Chalk_Quality', 0)) * 0.8)
        else:
            if row.get('Order', 0) == 0:
                proj *= 0.1
            elif row['Order'] <= 2:
                proj *= 1.25
            elif row['Order'] <= 5:
                proj *= 1.15

            if float(row.get('Edge_Value', 0)) > 0:
                proj = (proj * 0.5) + (row['Edge_Value'] * 0.15)

            b_h, o_h = row.get('CleanHand', '?'), p_hand_map.get(row['Opponent'], '?')
            if b_h == 'S' or (b_h == 'L' and o_h == 'R') or (b_h == 'R' and o_h == 'L'):
                proj *= 1.15

            wind, cond = str(w['wind']).lower(), str(w['condition']).lower()
            if "dome" not in cond:
                if isinstance(w['temp'], int) and w['temp'] >= 85: proj *= 1.10
                if 'out' in wind:
                    proj *= 1.08
                elif 'in' in wind:
                    proj *= 0.92

        # Apply randomness ONLY to unlocked games
        return proj * random.uniform(0.97, 1.03)

    # Calculate final solver projections and tag locked rows for the UI
    df['Solver_Proj'] = df.apply(apply_logic, axis=1)
    df['is_locked'] = df.apply(lambda r: game_statuses.get(" vs ".join(
        sorted([TEAM_MAP.get(str(r['Team']), str(r['Team'])), TEAM_MAP.get(str(r['Opponent']), str(r['Opponent']))])),
                                                           False), axis=1)

    teams_to_stack = [stack_team] if stack_team and stack_team != "None" else \
        df[~df['POS'].str.contains('P')].groupby('Team')['Solver_Proj'].mean().sort_values(ascending=False).head(
            8).index.tolist()

    for i in range(num_lineups):
        best_lineup, highest_score = None, -1
        random.shuffle(teams_to_stack)
        for current_team in teams_to_stack[:3]:
            try:
                prob = pulp.LpProblem(f"MLB_{i}_{current_team}", pulp.LpMaximize)
                players, slots = df.index.tolist(), list(POS_ORDER.keys())
                x = pulp.LpVariable.dicts("x", (players, slots), cat="Binary")

                # Objective
                prob += pulp.lpSum([df.loc[p, 'Solver_Proj'] * x[p][s] for p in players for s in slots])

                # Constraints
                prob += pulp.lpSum([df.loc[p, 'Salary'] * x[p][s] for p in players for s in slots]) <= 50000
                for s in slots: prob += pulp.lpSum([x[p][s] for p in players]) == 1
                for p in players:
                    prob += pulp.lpSum([x[p][s] for s in slots]) <= 1
                    if df.loc[p, 'Player'] in locks: prob += pulp.lpSum([x[p][s] for s in slots]) == 1
                    if player_usage.get(p, 0) >= max_count: prob += pulp.lpSum([x[p][s] for s in slots]) == 0

                    pos = str(df.loc[p, 'POS'])
                    for s in slots:
                        valid = any([(s.startswith('P') and 'P' in pos), (s == 'C' and 'C' in pos),
                                     (s == '1B' and '1B' in pos), (s == '2B' and '2B' in pos),
                                     (s == '3B' and '3B' in pos), (s == 'SS' and 'SS' in pos),
                                     (s.startswith('OF') and 'OF' in pos)])
                        if not valid: prob += x[p][s] == 0

                for past in used_player_indices:
                    prob += pulp.lpSum([x[p][s] for p in past for s in slots]) <= (len(slots) - diversity)

                h_idx = df[(df['Team'] == current_team) & (~df['POS'].str.contains('P'))].index.tolist()
                if len(h_idx) >= int(min_stack):
                    prob += pulp.lpSum([x[p][s] for p in h_idx for s in slots]) >= int(min_stack)

                prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=3))

                if pulp.LpStatus[prob.status] == 'Optimal':
                    score = pulp.value(prob.objective)
                    if score > highest_score:
                        highest_score = score
                        lineup_data, p_indices, t_sal, t_proj = [], [], 0, 0
                        for p in players:
                            for s in slots:
                                if pulp.value(x[p][s]) == 1:
                                    row = df.loc[p]
                                    # Inside the "if pulp.LpStatus[prob.status] == 'Optimal':" block
                                    lineup_data.append({
                                        'Slot': s,
                                        'Name': row['Player'],
                                        'Team': row['Team'],
                                        'Opponent': row['Opponent'],
                                        'Order': row.get('Order', 0),
                                        'Hand': row.get('CleanHand', '?'),
                                        'Logo': get_logo_url(row['Team']),
                                        'Proj': round(row['Solver_Proj'], 2),
                                        'Salary': row['Salary'],
                                        'W_Icon': row.get('W_Icon', ''),
                                        'Weather_Short': row.get('Weather_Short', ''),
                                        'Primary_Stat': row.get('Primary_Stat', ''),
                                        'is_locked': row.get('is_locked', False)
                                    })
                                    p_indices.append(p)
                                    t_sal += row['Salary']
                                    t_proj += row['Solver_Proj']

                        lineup_data.sort(key=lambda x: x['SortKey'])
                        best_lineup = {
                            'players': lineup_data,
                            'total_salary': t_sal,
                            'total_projection': round(t_proj, 2),
                            'indices': p_indices
                        }
            except:
                continue

        if best_lineup:
            all_results.append(best_lineup)
            used_player_indices.append(best_lineup['indices'])
            for idx in best_lineup['indices']: player_usage[idx] += 1
        else:
            break

    return all_results