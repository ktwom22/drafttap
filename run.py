import os
from flask import Flask, render_template, Response
from blueprints.routes_nba import nba_bp
from blueprints.routes_mlb import mlb_bp

# Initialize the Flask App
app = Flask(__name__)

# Register Blueprints with the correct URL prefixes
# This ensures /nba and /mlb routes work correctly
app.register_blueprint(nba_bp, url_prefix='/nba')
app.register_blueprint(mlb_bp, url_prefix='/mlb')


@app.route('/')
def index():
    """
    Main landing page for Draft Tap.
    Ensure 'index.html' is inside your /templates folder.
    """
    return render_template('index.html')


@app.route('/robots.txt')
def robots():
    """
    SEO/AISEO: Tells search engine bots they are allowed to crawl the site.
    """
    return Response("User-agent: *\nAllow: /", mimetype="text/plain")


@app.route('/sitemap.xml')
def sitemap():
    """
    SEO: Helps Google find all your optimizer pages quickly.
    """
    xml = """<?xml version="1.0" encoding="UTF-8"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
        <url><loc>https://web-production-f03c8.up.railway.app/</loc></url>
        <url><loc>https://web-production-f03c8.up.railway.app/nba</loc></url>
        <url><loc>https://web-production-f03c8.up.railway.app/mlb</loc></url>
    </urlset>"""
    return Response(xml, mimetype='application/xml')


if __name__ == "__main__":
    # Railway passes a PORT environment variable.
    # If not found (local testing), it defaults to 5000.
    port = int(os.environ.get("PORT", 5000))

    # Setting host to 0.0.0.0 is required for Railway to bind the service
    app.run(host='0.0.0.0', port=port, debug=False)