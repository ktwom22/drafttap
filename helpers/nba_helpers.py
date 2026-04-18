def get_nba_logo_url(team_abbrev):
    """
    Returns the high-res transparent logo URL from ESPN's CDN.
    """
    if not team_abbrev or str(team_abbrev).lower() == 'nan':
        return "https://www.draftkings.com/favicon.ico"  # Fallback

    # Standardize DraftKings/Common abbreviations to ESPN's format
    mapping = {
        'NY': 'NYK',
        'GS': 'GSW',
        'NO': 'NOP',
        'SA': 'SAS',
        'WSH': 'WAS',
        'PHX': 'PHO',
        'BKN': 'BKN',
        'CHA': 'CHA'
    }

    clean_abbrev = str(team_abbrev).upper().strip()
    espn_abbrev = mapping.get(clean_abbrev, clean_abbrev)

    return f"https://a.espncdn.com/i/teamlogos/nba/500/{espn_abbrev}.png"


def get_nba_matchup_info(df_row):
    """
    Optional: Helper to format the matchup string consistently.
    """
    return f"{df_row['Team']} @ {df_row['Opponent']}"