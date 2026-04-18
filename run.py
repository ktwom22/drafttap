from flask import Flask, render_template  # Make sure render_template is imported
from blueprints.routes_nba import nba_bp
from blueprints.routes_mlb import mlb_bp

app = Flask(__name__)

app.register_blueprint(nba_bp)
app.register_blueprint(mlb_bp)

@app.route('/')
def index():
    # This directly opens the index.html file in your /templates folder
    return render_template('index.html')

@app.app.route('/robots.txt')
def robots():
    return Response("User-agent: *\nAllow: /", mimetype="text/plain")

if __name__ == "__main__":
    app.run(debug=True)