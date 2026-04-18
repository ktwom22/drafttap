from flask import Flask, render_template  # Make sure render_template is imported
from blueprints.routes_nba import nba_bp
from blueprints.routes_mlb import mlb_bp

app = Flask(__name__)

app.register_blueprint(nba_bp)
app.register_blueprint(mlb_bp)

# CHANGE THIS:
# @app.app.route('/robots.txt')

# TO THIS:
@app.route('/robots.txt')
def robots():
    from flask import Response
    return Response("User-agent: *\nAllow: /", mimetype="text/plain")

@app.route('/sitemap.xml')
def sitemap():
    from flask import Response
    xml = """<?xml version="1.0" encoding="UTF-8"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
        <url><loc>https://web-production-f03c8.up.railway.app/</loc></url>
        <url><loc>https://web-production-f03c8.up.railway.app/nba</loc></url>
        <url><loc>https://web-production-f03c8.up.railway.app/mlb</loc></url>
    </urlset>"""
    return Response(xml, mimetype='application/xml')

if __name__ == "__main__":
    app.run(debug=True)