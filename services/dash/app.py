# app.py
import dash
from dash import html, dcc
import traceback
import os, sys, logging

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

external_stylesheets = [
    "https://cdn.jsdelivr.net/npm/bootswatch@5.1.3/dist/darkly/bootstrap.min.css"
]

app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=external_stylesheets,
    suppress_callback_exceptions=True,
    update_title=None,
)
app.title = "GeoHydroAI | DEM OLAP"
server = app.server

# ВАЖЛИВО: реєструємо всі колбеки; якщо впаде — покажемо трейс і все одно стартуємо
try:
    import callbacks.main_callbacks  # noqa: F401
    # import callbacks.dem_diff_callbacks
    import callbacks.sidebar_drawer
    import callbacks.cdf_callback
    import callbacks.best_model_callback
    import callbacks.map_profile_callback
except Exception as e:
    print("FATAL: failed to import callbacks.main_callbacks:", e)
    traceback.print_exc()

def _log_callbacks_once():
    logging.getLogger(__name__).info("=== Registered callbacks ===")
    for k in app.callback_map.keys():
        logging.getLogger(__name__).info(" - %s", k)
    logging.getLogger(__name__).info("=== End callbacks ===")

_log_callbacks_once()

navbar = html.Div(
    [
        dcc.Link("Dashboard", href="/", style={"color": "#00bfff", "marginLeft": "34px", "marginRight": "24px"}),
        dcc.Link("DEM Diff Analysis", href="/dem-diff", style={"color": "#00bfff", "marginRight": "24px"}),
        dcc.Link("DEM Diff Analysis-1", href="/dem-diff-1", style={"color": "#00bfff", "marginRight": "24px"}),
        dcc.Link("ICESat-2 Map", href="/icesat-map", style={"color": "#00bfff", "marginRight": "24px"}),
        dcc.Link("Best DEM", href="/best-dem", style={"color": "#00bfff", "marginRight": "24px"}),
        dcc.Link("CDF Accumulation", href="/cdf", style={"color": "#00bfff", "marginRight": "24px"}),
        dcc.Link("Flood Scenarios Test", href="/flood-test", style={"color": "#00bfff", "marginRight": "24px"}),
    ],
    style={"padding": "20px 0", "position": "sticky", "top": "0", "zIndex": 1100},
)

app.layout = html.Div([navbar, dash.page_container])

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)
