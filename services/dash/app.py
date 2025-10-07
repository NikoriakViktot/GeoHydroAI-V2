# app.py
import dash
from dash import html, dcc
import traceback
import os, sys, logging
import dash_deckgl
import logging
import dash_bootstrap_components as dbc

BASE_PATH = os.getenv("BASE_PATH", "/dem/")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,  # критично: перезаписати конфіг, який міг виставити gunicorn
)
if not BASE_PATH.startswith("/"):
    BASE_PATH = "/" + BASE_PATH
if not BASE_PATH.endswith("/"):
    BASE_PATH += "/"

logging.info("Dash BASE_PATH = %r", BASE_PATH)
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
    requests_pathname_prefix=BASE_PATH,
    routes_pathname_prefix=BASE_PATH,
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



@server.route("/health")
def health():
    return "ok", 200

# ІМПОРТИ КОЛБЕКІВ — без navigate
for mod in (
    "callbacks.main_callbacks",
    "callbacks.sidebar_drawer",
    "callbacks.cdf_callback",
    "callbacks.best_model_callback",
    "callbacks.map_profile_callback",
    "callbacks.profile_callback",
):
    try:
        __import__(mod)
    except Exception:
        logging.exception("FATAL: failed to import %s", mod)
        raise

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Dashboard", href=url("/dashboard"))),
        dbc.NavItem(dbc.NavLink("DEM Difference", href=url("/dem-diff"))),
        dbc.NavItem(dbc.NavLink("Flood Scenarios (Map)", href=url("/flood-dem-diif"))),
        dbc.NavItem(dbc.NavLink("FFA Report", href="https://geohydroai.org/reports/ffa_report_en.html", target="_blank")),
        dbc.NavItem(dbc.NavLink("Cross Section", href="https://geohydroai.org/reports/cross_section_dashboard.html", target="_blank")),
    ],
    brand="GeoHydroAI",
    color="dark",
    dark=True,
    sticky="top",
    class_name="py-1 gh-navbar",
)


app.layout = html.Div([dcc.Location(id="url"), navbar, dash.page_container])

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)