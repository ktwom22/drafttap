import statsapi
import pandas as pd
import time
import re
import unicodedata
import requests
import pytz
from datetime import datetime

# --- CACHE CONTROL ---
_STATS_CACHE = {'h': None, 'p': None, 'time': 0}

TEAM_MAP = {
    "CHW": "CWS", "CHA": "CWS", "CWS": "CWS", "WSH": "WAS", "WAS": "WAS", "Washington": "WAS",
    "OAK": "OAK", "ATH": "OAK", "SF": "SF", "SFO": "SF", "SFG": "SF", "San Francisco": "SF",
    "AZ": "ARI", "ARI": "ARI", "Arizona": "ARI", "TB": "TB", "TBA": "TB", "Tampa Bay": "TB",
    "KC": "KC", "KCA": "KC", "Kansas City": "KC", "SD": "SD", "SDN": "SD", "San Diego": "SD",
    "NYY": "NYY", "NYA": "NYY", "New York": "NYM", "NYN": "NYM",
    "LAD": "LAD", "LAN": "LAD", "Los Angeles": "LAD", "STL": "STL", "SLN": "STL", "St. Louis": "STL",
    "CHC": "CHC", "CHN": "CHC", "Chicago": "CHC", "TOR": "TOR", "Toronto": "TOR",
    "COL": "COL", "Colorado": "COL", "ATL": "ATL", "Atlanta": "ATL", "Boston": "BOS",
    "Miami": "MIA", "Philadelphia": "PHI", "Cleveland": "CLE", "Detroit": "DET",
    "Houston": "HOU", "Milwaukee": "MIL", "Minnesota": "MIN", "Pittsburgh": "PIT",
    "Seattle": "SEA", "Texas": "TEX", "Baltimore": "BAL", "Cincinnati": "CIN", "LAA": "LAA", "Anaheim": "LAA"
}

TEAM_ID_MAP = {
    "ARI": 109, "ATL": 144, "BAL": 110, "BOS": 111, "CHC": 112, "CWS": 145,
    "CIN": 113, "CLE": 114, "COL": 115, "DET": 116, "HOU": 117, "KC": 118,
    "LAA": 108, "LAD": 119, "MIA": 146, "MIL": 158, "MIN": 142, "NYM": 121,
    "NYY": 147, "OAK": 133, "PHI": 143, "PIT": 134, "SD": 135, "SF": 137,
    "SEA": 136, "STL": 138, "TB": 139, "TEX": 140, "TOR": 141, "WAS": 120
}


def clean_float(val, default=0.0):
    if val is None or str(val).strip() in ['-.--', '---', '-', '']: return default
    try:
        return float(val)
    except:
        return default


def normalize_name(name):
    if not isinstance(name, str): return ""
    name = unicodedata.normalize('NFD', name).encode('ascii', 'ignore').decode("utf-8")
    name = re.sub(r'\b(jr|sr|ii|iii|iv)\b', '', name, flags=re.IGNORECASE)
    return "".join(filter(str.isalnum, name)).lower()


def get_logo_url(team_abbr):
    clean_abbr = TEAM_MAP.get(team_abbr, team_abbr)
    tid = TEAM_ID_MAP.get(clean_abbr)
    return f"https://www.mlbstatic.com/team-logos/team-cap-on-light/{tid}.svg" if tid else "https://www.mlbstatic.com/team-logos/league/1.svg"


def fetch_pitcher_metrics(year):
    try:
        data = statsapi.get('stats', {'stats': 'season', 'season': year, 'group': 'pitching', 'playerPool': 'all',
                                      'limit': 1500})
        year_data = {}
        for s in data['stats'][0].get('splits', []):
            st, name = s.get('stat', {}), s.get('player', {}).get('fullName', '')
            if not name: continue
            year_data[normalize_name(name)] = {
                'full_name': name, 'WHIP': clean_float(st.get('whip'), 1.35),
                'K9': clean_float(st.get('strikeoutsPer9Inn'), 7.5), 'HR9': clean_float(st.get('homeRunsPer9'), 1.2),
                'K_BB': clean_float(st.get('strikeoutWalkRatio'), 2.5), 'GS': int(st.get('gamesStarted', 0))
            }
        return year_data
    except:
        return {}


def fetch_hitter_metrics(year):
    try:
        data = statsapi.get('stats',
                            {'stats': 'season', 'season': year, 'group': 'hitting', 'playerPool': 'all', 'limit': 1500})
        year_data = {}
        for s in data['stats'][0].get('splits', []):
            st, name = s.get('stat', {}), s.get('player', {}).get('fullName', '')
            if not name: continue
            ops = clean_float(st.get('ops'), 0.700)
            year_data[normalize_name(name)] = {
                'full_name': name, 'ISO': clean_float(st.get('slg'), 0.400) - clean_float(st.get('avg'), 0.250),
                'wRC_proxy': int((ops / 0.750) * 100)
            }
        return year_data
    except:
        return {}


def get_weighted_stats():
    global _STATS_CACHE
    if _STATS_CACHE['h'] is not None and (time.time() - _STATS_CACHE['time']) < 3600:
        return _STATS_CACHE['h'], _STATS_CACHE['p']

    W_PREV, W_CURR = 0.7, 0.3
    p25, p26 = fetch_pitcher_metrics(2025), fetch_pitcher_metrics(2026)
    blended_p = []
    for name in set(p25.keys()) | set(p26.keys()):
        s25, s26 = p25.get(name), p26.get(name)
        blend = lambda k, d: (s25[k] * W_PREV + s26[k] * W_CURR) if (s25 and s26 and k in s25 and k in s26) else (
            s26[k] if s26 else (s25[k] if s25 else d))
        k9, kbb, whip = blend('K9', 7.5), blend('K_BB', 2.5), blend('WHIP', 1.35)
        blended_p.append({
            'norm_name': name, 'full_name': s26['full_name'] if s26 else s25['full_name'],
            'Chalk_Quality': round(max(0.1, (k9 * 0.5 + kbb * 0.5 - whip * 2.0)), 2), 'WHIP': round(whip, 2),
            'K/9': round(k9, 2), 'HR/9': blend('HR9', 1.2), 'GS': s26['GS'] if s26 else (s25['GS'] if s25 else 0)
        })

    h25, h26 = fetch_hitter_metrics(2025), fetch_hitter_metrics(2026)
    blended_h = []
    for name in set(h25.keys()) | set(h26.keys()):
        s25, s26 = h25.get(name), h26.get(name)
        blend = lambda k, d: (s25[k] * W_PREV + s26[k] * W_CURR) if (s25 and s26 and k in s25 and k in s26) else (
            s26[k] if s26 else (s25[k] if s25 else d))
        iso = blend('ISO', 0.150)
        blended_h.append({
            'norm_name': name, 'full_name': s26['full_name'] if s26 else s25['full_name'],
            'ISO': round(iso, 3), 'wRC+': int(blend('wRC_proxy', 100)), 'Edge_Value': round(iso * 400, 2)
        })

    h_df, p_df = pd.DataFrame(blended_h), pd.DataFrame(blended_p)
    _STATS_CACHE.update({'h': h_df, 'p': p_df, 'time': time.time()})
    return h_df, p_df


def get_mlb_weather_data():
    weather_map = {}
    api_to_abbr = {"Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL",
                   "Boston Red Sox": "BOS", "Chicago Cubs": "CHC", "Chicago White Sox": "CWS", "Cincinnati Reds": "CIN",
                   "Cleveland Guardians": "CLE", "Colorado Rockies": "COL", "Detroit Tigers": "DET",
                   "Houston Astros": "HOU", "Kansas City Royals": "KC", "Los Angeles Angels": "LAA",
                   "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL",
                   "Minnesota Twins": "MIN", "New York Mets": "NYM", "New York Yankees": "NYY",
                   "Oakland Athletics": "OAK", "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT",
                   "San Diego Padres": "SD", "San Francisco Giants": "SF", "Seattle Mariners": "SEA",
                   "St. Louis Cardinals": "STL", "Tampa Bay Rays": "TB", "Texas Rangers": "TEX",
                   "Toronto Blue Jays": "TOR", "Washington Nationals": "WAS"}
    try:
        games = statsapi.schedule(date=datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d'))
        for g in games:
            a, h = api_to_abbr.get(g['away_name']), api_to_abbr.get(g['home_name'])
            if not a or not h: continue
            # Lookup key is alphabetical for internal matching
            gid = " vs ".join(sorted([a, h]))
            det = statsapi.get('game', {'gamePk': g['game_id']})
            w = det.get('gameData', {}).get('weather', {})
            weather_map[gid] = {
                'temp': int(w.get('temp', 70)) if str(w.get('temp')).isdigit() else 70,
                'wind': w.get('wind', '0 mph, Calm'),
                'condition': w.get('condition', 'Clear'),
                'away': a,
                'home': h
            }
    except:
        pass
    return weather_map


def get_espn_game_info():
    """
    Returns a mapping of lookup_id (alphabetical) to real game details.
    This uses the logic from your audit to identify Home vs Away correctly.
    """
    date_str = datetime.now(pytz.timezone('US/Eastern')).strftime('%Y%m%d')
    url = f"https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates={date_str}"
    game_info = {}
    try:
        data = requests.get(url).json()
        for e in data.get('events', []):
            comp = e['competitions'][0]['competitors']

            # Explicitly identify Away and Home using the homeAway key
            home_raw = next(t['team']['abbreviation'] for t in comp if t['homeAway'] == 'home')
            away_raw = next(t['team']['abbreviation'] for t in comp if t['homeAway'] == 'away')

            h = TEAM_MAP.get(home_raw, home_raw)
            a = TEAM_MAP.get(away_raw, away_raw)

            # Alphabetical key for the lookup logic (consistent with weather)
            lookup_id = " vs ".join(sorted([h, a]))

            utc = datetime.strptime(e['date'], "%Y-%m-%dT%H:%MZ").replace(tzinfo=pytz.utc)
            game_info[lookup_id] = {
                'display': f"{a} @ {h}",
                'raw_time': utc,
                'home_team': h,
                'away_team': a,
                'time_str': utc.astimezone(pytz.timezone('US/Eastern')).strftime('%I:%M %p')
            }
    except:
        pass
    return game_info