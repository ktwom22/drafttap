import os
from flask import Flask, render_template, Response
from flask_caching import Cache
from blueprints.routes_nba import nba_bp
from blueprints.routes_mlb import mlb_bp
from blueprints.routes_nhl import nhl_bp

# Import your slug generator for SEO
from helpers.mlb_helpers import get_all_matchup_slugs

app = Flask(__name__)

# --- CACHE CONFIGURATION ---
REDIS_URL = os.environ.get('REDIS_URL')

if REDIS_URL:
    # Use Redis on Railway
    app.config['CACHE_TYPE'] = 'RedisCache'
    app.config['CACHE_REDIS_URL'] = REDIS_URL
    print("Caching Status: REDIS CONNECTED")
else:
    # Use local memory when testing in PyCharm
    app.config['CACHE_TYPE'] = 'SimpleCache'
    print("Caching Status: LOCAL MEMORY (Redis not found)")

app.config['CACHE_DEFAULT_TIMEOUT'] = 300
cache = Cache(app)
app.cache = cache

# Register Blueprints
app.register_blueprint(nba_bp, url_prefix='/nba')
app.register_blueprint(mlb_bp, url_prefix='/mlb')
app.register_blueprint(nhl_bp, url_prefix='/nhl')


@app.route('/')
@cache.cached(timeout=600)
def index():
    return render_template('index.html')


@app.route('/robots.txt')
def robots():
    # Adding the Sitemap link to robots.txt helps Google find it faster
    return Response("User-agent: *\nAllow: /\nSitemap: https://drafttap.com/sitemap.xml", mimetype="text/plain")


@app.route('/sitemap.xml')
def sitemap():
    """
    DYNAMIC SEO ENGINE:
    This generates a fresh link for every single matchup today.
    """
    base_url = "https://drafttap.com"

    # Get all today's MLB matchups (and eventually NBA/NHL)
    # This turns into list like: ['mlb/nyy-vs-bos', 'mlb/lad-vs-sd']
    matchup_slugs = get_all_matchup_slugs()

    xml = '<?xml version="1.0" encoding="UTF-8"?>'
    xml += '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'

    # 1. Main Hubs
    for route in ['', '/nba', '/mlb', '/nhl']:
        xml += f'<url><loc>{base_url}{route}</loc><changefreq>daily</changefreq><priority>1.0</priority></url>'

    # 2. Dynamic Matchup Pages (The Traffic Driver)
    for slug in matchup_slugs:
        xml += f'<url><loc>{base_url}/{slug}</loc><changefreq>hourly</changefreq><priority>0.8</priority></url>'

    xml += '</urlset>'
    return Response(xml, mimetype='application/xml')


if __name__ == "__main__":
    # Ensure port is pulled from environment for Railway
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)