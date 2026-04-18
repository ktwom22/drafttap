import os


class Config:
    # --- APP SETTINGS ---
    SECRET_KEY = os.environ.get('SECRET_KEY', 'betify-pro-secret-key-2026')
    DEBUG = True

    # --- DATA SOURCES ---
    # Current MLB Salary/Projection Sheet
    MLB_SALARY_CSV = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRzCRSTDnslz-zmGESH1CFhjsYD7NJa8yHkapMFu1JIR0M1PQDwZzMIDCmhPBUNU6kzLJy8-3_ioR4Y/pub?gid=1189680617&single=true&output=csv"

    # Placeholder for NBA Sheet (Swap this when your NBA sheet is ready)
    NBA_SALARY_CSV = "https://docs.google.com/spreadsheets/d/YOUR_NBA_SHEET_HERE/pub?output=csv"

    # --- SPORT SPECIFIC CONFIGS ---
    MLB_CONFIG = {
        'name': 'MLB',
        'cap': 50000,
        'pos_order': {
            'P1': 0, 'P2': 1, 'C': 2, '1B': 3, '2B': 4,
            '3B': 5, 'SS': 6, 'OF1': 7, 'OF2': 8, 'OF3': 9
        }
    }

    NBA_CONFIG = {
        'name': 'NBA',
        'cap': 50000,
        'pos_order': {
            'PG': 0, 'SG': 1, 'SF': 2, 'PF': 3,
            'C': 4, 'G': 5, 'F': 6, 'UTIL': 7
        }
    }

    # --- TEAM MAPPING (Shared across scrapers) ---
    TEAM_MAP = {
        "CHW": "CWS", "CHA": "CWS", "CWS": "CWS", "WSH": "WAS", "WAS": "WAS",
        "Washington": "WAS", "OAK": "OAK", "ATH": "OAK", "SF": "SF", "SFO": "SF",
        "SFG": "SF", "San Francisco": "SF", "AZ": "ARI", "ARI": "ARI",
        "Arizona": "ARI", "TB": "TB", "TBA": "TB", "Tampa Bay": "TB",
        "KC": "KC", "KCA": "KC", "Kansas City": "KC", "SD": "SD", "SDN": "SD",
        "San Diego": "SD", "NYY": "NYY", "NYA": "NYY", "NYM": "NYM", "NYN": "NYM",
        "LAD": "LAD", "LAN": "LAD", "Los Angeles": "LAD", "STL": "STL", "SLN": "STL",
        "St. Louis": "STL", "CHC": "CHC", "CHN": "CHC", "Chicago": "CHC", "TOR": "TOR",
        "Toronto": "TOR", "COL": "COL", "Colorado": "COL", "ATL": "ATL", "Atlanta": "ATL",
        "Boston": "BOS", "Miami": "MIA", "Philadelphia": "PHI", "Cleveland": "CLE",
        "Detroit": "DET", "Houston": "HOU", "Milwaukee": "MIL", "Minnesota": "MIN",
        "Pittsburgh": "PIT", "Seattle": "SEA", "Texas": "TEX", "Baltimore": "BAL",
        "Cincinnati": "CIN", "LAA": "LAA", "Anaheim": "LAA"
    }

    TEAM_ID_MAP = {
        "ARI": 109, "ATL": 144, "BAL": 110, "BOS": 111, "CHC": 112, "CWS": 145,
        "CIN": 113, "CLE": 114, "COL": 115, "DET": 116, "HOU": 117, "KC": 118,
        "LAA": 108, "LAD": 119, "MIA": 146, "MIL": 158, "MIN": 142, "NYM": 121,
        "NYY": 147, "OAK": 133, "PHI": 143, "PIT": 134, "SD": 135, "SF": 137,
        "SEA": 136, "STL": 138, "TB": 139, "TEX": 140, "TOR": 141, "WAS": 120
    }

    # --- NBA TEAM ID MAP (Optional, for NBA logos later) ---
    NBA_TEAM_ID_MAP = {
        "BOS": 1610612738, "GSW": 1610612744, "LAL": 1610612747,  # Add others as needed
    }