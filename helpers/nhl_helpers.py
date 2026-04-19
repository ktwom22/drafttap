import requests
from rapidfuzz import process  # pip install rapidfuzz


def get_dynamic_nhl_ids():
    """
    Fetches all active NHL players from ESPN's API and returns a name-to-id map.
    Uses a session for faster connection pooling.
    """
    id_map = {}
    session = requests.Session()
    try:
        # Step 1: Get all NHL teams in one shot
        teams_url = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/teams?limit=100"
        teams_response = session.get(teams_url, timeout=5)
        teams_data = teams_response.json()

        teams_list = teams_data.get('sports', [{}])[0].get('leagues', [{}])[0].get('teams', [])

        for team_entry in teams_list:
            team_id = team_entry['team']['id']
            # Step 2: Fetch the roster for each team
            roster_url = f"https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/teams/{team_id}/roster"
            roster_response = session.get(roster_url, timeout=5)
            roster_data = roster_response.json()

            # Step 3: Map athlete names to their IDs
            for athlete in roster_data.get('athletes', []):
                name = athlete.get('displayName')
                athlete_id = athlete.get('id')
                if name and athlete_id:
                    id_map[name] = str(athlete_id)

        print(f"--- SUCCESS: Mapped {len(id_map)} NHL Headshots ---")
    except Exception as e:
        print(f"Error fetching dynamic NHL IDs: {e}")
    return id_map


def get_nhl_headshot_url(player_name, id_map):
    """
    Tries exact match first, then fuzzy match, then fallback.
    """
    if not player_name or str(player_name).lower() == 'nan':
        return "https://a.espncdn.com/i/headshots/nhl/players/full/0.png"

    name_str = str(player_name).strip()

    # 1. Try Exact Match
    if name_str in id_map:
        return f"https://a.espncdn.com/i/headshots/nhl/players/full/{id_map[name_str]}.png"

    # 2. Try Fuzzy Match (85% similarity threshold)
    # This helps with "Alex" vs "Alexander" or "N. MacKinnon" vs "Nathan MacKinnon"
    all_names = list(id_map.keys())
    if all_names:
        match = process.extractOne(name_str, all_names, score_cutoff=85)
        if match:
            matched_name = match[0]
            return f"https://a.espncdn.com/i/headshots/nhl/players/full/{id_map[matched_name]}.png"

    # 3. Fallback to silhouette
    return "https://a.espncdn.com/i/headshots/nhl/players/full/0.png"


def get_nhl_logo_url(team_abbrev):
    """
    Returns the high-res transparent logo URL from ESPN.
    """
    if not team_abbrev or str(team_abbrev).lower() == 'nan':
        return "https://www.draftkings.com/favicon.ico"

    # Standard DFS -> ESPN Abbreviation Mapping
    mapping = {
        'SJ': 'SJS', 'TB': 'TBL', 'LA': 'LAK', 'NJ': 'NJD',
        'WAS': 'WSH', 'CBJ': 'CLB', 'NAS': 'NSH', 'VGS': 'VGK',
        'MON': 'MTL', 'OTT': 'OTT', 'CAL': 'CGY'
    }

    clean_abbrev = str(team_abbrev).upper().strip()
    espn_abbrev = mapping.get(clean_abbrev, clean_abbrev)

    return f"https://a.espncdn.com/i/teamlogos/nhl/500/{espn_abbrev}.png"


def get_nhl_matchup_info(df_row):
    """
    Returns a consistent AWAY @ HOME string based on row data.
    """
    # Ensure we are looking at strings
    team = str(df_row.get('Team', 'TBD')).upper().strip()
    opp = str(df_row.get('Opponent', 'TBD')).upper().strip()

    # Check for home/away indicators in the sheet
    location = str(df_row.get('LOCATION', '')).lower()
    is_home = df_row.get('IS_HOME', None)

    if location == 'home' or is_home is True:
        return f"{opp} @ {team}"
    elif location == 'away' or is_home is False:
        return f"{team} @ {opp}"

    # Alphabetical fallback if no location data exists
    teams = sorted([team, opp])
    return f"{teams[0]} @ {teams[1]}"