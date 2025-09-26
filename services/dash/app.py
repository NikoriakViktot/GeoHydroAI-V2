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
    requests_pathname_prefix='/dem/',
    use_pages=True,
    external_stylesheets=[
        "https://cdn.jsdelivr.net/npm/bootswatch@5.1.3/dist/darkly/bootstrap.min.css"
    ],
    suppress_callback_exceptions=True,
    update_title=None,
)
app.title = "GeoHydroAI | DEM OLAP"
server = app.server
def url(p: str) -> str:
    return app.get_relative_path(p)

# from flask import Response
# @server.route("/_dash-component-suites/dash_deckgl/dash_deckgl.min.js.map")
# def _dash_deckgl_sourcemap():
#     return Response("{}", mimetype="application/json")

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

from flask import abort, request
# server = app.server
@server.before_request
def block_non_dem():
    p = request.path
    if not (p.startswith("/dem/")
            or p.startswith("/_dash")
            or p.startswith("/assets")):
        abort(404)



navbar = html.Div(
    [
        dcc.Link("Dashboard ", href="/dashboard", className="btn btn-primary", style={"marginRight": "28px"}),
        dcc.Link("DEM Difference", href="/dem-diff", className="btn btn-secondary", style={"marginRight": "28px"}),
        dcc.Link("Flood Scenarios", href="/flood-dem-diif", className="btn btn-secondary", style={"marginRight": "28px"}),
    ],
    style={
        "padding": "10px 20px",
        "position": "sticky",
        "top": "0",
        "zIndex": 1100,
        "backgroundColor": "#222",
        "width": "100%",
    },
)


app.layout = html.Div([dcc.Location(id="url"), navbar, dash.page_container])

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)