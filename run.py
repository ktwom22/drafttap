import os
from flask import Flask, render_template, Response
from flask_caching import Cache  # New Import
from blueprints.routes_nba import nba_bp
from blueprints.routes_mlb import mlb_bp
from blueprints.routes_nhl import nhl_bp

# Initialize the Flask App
app = Flask(__name__)

# --- CACHE CONFIGURATION ---
# Railway provides REDIS_URL automatically when you add the Redis service
app.config['CACHE_TYPE'] = 'RedisCache'
app.config['CACHE_REDIS_URL'] = os.environ.get('REDIS_URL', 'redis://localhost:6379')
app.config['CACHE_DEFAULT_TIMEOUT'] = 300  # 5 Minute default pour

# Global cache object
cache = Cache(app)

# Attach cache to the app so blueprints can access it via current_app
app.cache = cache
# ---------------------------

# Register Blueprints
app.register_blueprint(nba_bp, url_prefix='/nba')
app.register_blueprint(mlb_bp, url_prefix='/mlb')
app.register_blueprint(nhl_bp, url_prefix='/nhl')

@app.route('/')
@cache.cached(timeout=600) # Index is static mostly, cache for 10 mins
def index():
    return render_template('index.html')

@app.route('/robots.txt')
def robots():
    return Response("User-agent: *\nAllow: /", mimetype="text/plain")

@app.route('/sitemap.xml')
def sitemap():
    xml = """<?xml version="1.0" encoding="UTF-8"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
        <url><loc>https://draftap.com/</loc></url>
        <url><loc>https://draftap.com/nba</loc></url>
        <url><loc>https://draftap.com/mlb</loc></url>
        <url><loc>https://draftap.com/nhl</loc></url>
    </urlset>"""
    return Response(xml, mimetype='application/xml')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)