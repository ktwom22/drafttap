import requests


def get_dynamic_nhl_ids():
    """
    Fetches all active NHL players from ESPN's API and returns a name-to-id map.
    Essential for mapping cleaned player names to headshot IDs.
    """
    id_map = {}
    try:
        # Step 1: Get all NHL teams
        teams_url = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/teams?limit=100"
        teams_response = requests.get(teams_url, timeout=5)
        teams_data = teams_response.json()

        teams_list = teams_data.get('sports', [{}])[0].get('leagues', [{}])[0].get('teams', [])

        for team_entry in teams_list:
            team_id = team_entry['team']['id']
            # Step 2: Fetch the roster for each team
            roster_url = f"https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/teams/{team_id}/roster"
            roster_response = requests.get(roster_url, timeout=5)
            roster_data = roster_response.json()

            # Step 3: Map athlete names to their IDs
            for athlete in roster_data.get('athletes', []):
                name = athlete.get('displayName')
                athlete_id = athlete.get('id')
                if name and athlete_id:
                    id_map[name] = str(athlete_id)

    except Exception as e:
        print(f"Error fetching dynamic NHL IDs: {e}")

    return id_map


def get_nhl_logo_url(team_abbrev):
    """
    Returns the high-res transparent logo URL from ESPN's CDN for NHL.
    """
    if not team_abbrev or str(team_abbrev).lower() == 'nan':
        return "https://www.draftkings.com/favicon.ico"

    # Common discrepancies between DK and ESPN abbreviations
    mapping = {
        'VEG': 'VGK',  # Vegas Golden Knights
        'WAS': 'WSH',  # Washington Capitals
        'SJ': 'SJS',  # San Jose Sharks
        'NJ': 'NJD',  # New Jersey Devils
        'TB': 'TBL',  # Tampa Bay Lightning
        'LA': 'LAK',  # Los Angeles Kings
        'OTT': 'OTT',
        'MON': 'MTL',  # Montreal Canadiens
        'CLB': 'CBJ'  # Columbus Blue Jackets
    }

    clean_abbrev = str(team_abbrev).upper().strip()
    espn_abbrev = mapping.get(clean_abbrev, clean_abbrev)

    return f"https://a.espncdn.com/i/teamlogos/nhl/500/{espn_abbrev}.png"


def get_nhl_matchup_info(df_row):
    """
    Returns a consistent AWAY @ HOME string for NHL.
    """
    team = str(df_row.get('Team', '')).upper()
    opp = str(df_row.get('Opponent', '')).upper()

    # If your CSV uses the '@' or 'vs' indicator in a 'Matchup' column
    matchup_raw = str(df_row.get('Matchup', ''))

    if '@' in matchup_raw:
        return matchup_raw  # Use the provided matchup if it exists

    # Fallback to alphabetical sorting to ensure consistency across both team perspectives
    teams = sorted([team, opp])
    return f"{teams[0]} @ {teams[1]}"


def get_nhl_headshot_url(player_name, id_map):
    """
    Look up the player ID from the dynamic map and return the ESPN headshot URL.
    Defaults to 0.png (silhouette) if name not found.
    """
    player_id = id_map.get(player_name, "0")

    try:
        if not player_id or str(player_id) == '0':
            return "https://a.espncdn.com/i/headshots/nhl/players/full/0.png"

        return f"https://a.espncdn.com/i/headshots/nhl/players/full/{player_id}.png"
    except Exception:
        return "https://a.espncdn.com/i/headshots/nhl/players/full/0.png"