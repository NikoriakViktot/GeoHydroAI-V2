# app.py
import dash
from dash import html, dcc
import traceback
import os, sys, logging
import dash_deckgl
import logging

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,  # критично: перезаписати конфіг, який міг виставити gunicorn
)

# Додатково: вирівняти Flask/Dash під gunicorn, якщо gunicorn вже має свої хендлери
gunicorn_error = logging.getLogger("gunicorn.error")
if gunicorn_error.handlers:
    root = logging.getLogger()
    root.handlers = gunicorn_error.handlers
    root.setLevel(gunicorn_error.level)

# external_stylesheets = [
#     "https://cdn.jsdelivr.net/npm/bootswatch@5.1.3/dist/darkly/bootstrap.min.css"
# ]
app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[
        "https://cdn.jsdelivr.net/npm/bootswatch@5.1.3/dist/darkly/bootstrap.min.css"
    ],
    suppress_callback_exceptions=True,
    update_title=None,
)
app.title = "GeoHydroAI | DEM OLAP"
server = app.server

from flask import Response
@server.route("/_dash-component-suites/dash_deckgl/dash_deckgl.min.js.map")
def _dash_deckgl_sourcemap():
    return Response("{}", mimetype="application/json")

# ІМПОРТИ КОЛБЕКІВ — без navigate
for mod in (
    "callbacks.main_callbacks",
    "callbacks.sidebar_drawer",
    "callbacks.cdf_callback",
    "callbacks.best_model_callback",
    # "callbacks.map_profile_callback",
    "callbacks.profile_callback",
):
    try:
        __import__(mod)
    except Exception:
        logging.exception("FATAL: failed to import %s", mod)



navbar = html.Div([
    dcc.Link("Dashboard", href="/", className="btn btn-primary", style={"marginRight": "28px"}),
    dcc.Link("DEM Diff", href="/dem-diff", className="btn btn-secondary", style={"marginRight": "28px"}),
    html.A("Flood Scenarios", href="/flood_scenarios/", className="btn btn-secondary"),  # <- зовнішнє
], style={"padding":"20px 0","position":"sticky","top":"0","zIndex":1100})

app.layout = html.Div([dcc.Location(id="url"), navbar, dash.page_container])

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)