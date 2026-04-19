import requests


def get_dynamic_espn_ids():
    """
    Fetches all active NBA players from ESPN's API and returns a name-to-id map.
    This prevents broken images when the CSV is missing ID data.
    """
    id_map = {}
    try:
        # Step 1: Get all NBA teams
        teams_url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams?limit=100"
        teams_response = requests.get(teams_url, timeout=5)
        teams_data = teams_response.json()

        # Get the list of teams
        teams_list = teams_data.get('sports', [{}])[0].get('leagues', [{}])[0].get('teams', [])

        for team_entry in teams_list:
            team_id = team_entry['team']['id']
            # Step 2: Fetch the roster for each team
            roster_url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/roster"
            roster_response = requests.get(roster_url, timeout=5)
            roster_data = roster_response.json()

            # Step 3: Map athlete names to their IDs
            for athlete in roster_data.get('athletes', []):
                # We store the name as key for quick lookup
                name = athlete.get('displayName')
                athlete_id = athlete.get('id')
                if name and athlete_id:
                    id_map[name] = str(athlete_id)

    except Exception as e:
        print(f"Error fetching dynamic ESPN IDs: {e}")

    return id_map


def get_nba_logo_url(team_abbrev):
    """
    Returns the high-res transparent logo URL from ESPN's CDN.
    Standardized for NBA abbreviations.
    """
    if not team_abbrev or str(team_abbrev).lower() == 'nan':
        return "https://www.draftkings.com/favicon.ico"

    mapping = {
        'NY': 'NYK', 'GS': 'GSW', 'NO': 'NOP', 'SA': 'SAS',
        'WSH': 'WAS', 'PHX': 'PHO', 'BKN': 'BKN', 'CHA': 'CHA'
    }

    clean_abbrev = str(team_abbrev).upper().strip()
    espn_abbrev = mapping.get(clean_abbrev, clean_abbrev)

    return f"https://a.espncdn.com/i/teamlogos/nba/500/{espn_abbrev}.png"


def get_nba_matchup_info(df_row):
    """
    Returns a consistent AWAY @ HOME string.
    Ensures that players on both teams show the same matchup string.
    """
    team = str(df_row.get('Team', '')).upper()
    opp = str(df_row.get('Opponent', '')).upper()

    location = str(df_row.get('Location', '')).lower()
    is_home = df_row.get('Is_Home', False)

    if location == 'home' or is_home is True:
        return f"{opp} @ {team}"
    elif location == 'away' or is_home is False:
        return f"{team} @ {opp}"

    # Fallback to prevent BOS@PHI vs PHI@BOS issue
    teams = sorted([team, opp])
    return f"{teams[0]} @ {teams[1]}"


def get_player_headshot_url(player_id):
    """
    Returns the high-res transparent headshot from ESPN.
    Defaults to ID 0 (silhouette) if ID is invalid.
    """
    try:
        # Handle cases where player_id is None, nan, or empty string
        if not player_id or str(player_id).lower() == 'nan' or str(player_id) == '0':
            return "https://a.espncdn.com/i/headshots/nba/players/full/0.png"

        # Strip decimals (e.g., '12345.0')
        clean_id = str(int(float(player_id)))
        return f"https://a.espncdn.com/i/headshots/nba/players/full/{clean_id}.png"
    except (ValueError, TypeError):
        return "https://a.espncdn.com/i/headshots/nba/players/full/0.png"