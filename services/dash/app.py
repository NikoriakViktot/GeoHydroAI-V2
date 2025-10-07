import dash
from dash import html, dcc
import os, sys, logging
import dash_deckgl
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

BASE_PATH = os.getenv("BASE_PATH", "/dem/")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
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

app = dash.Dash(
    __name__,
    requests_pathname_prefix=BASE_PATH,
    routes_pathname_prefix=BASE_PATH,
    use_pages=True,
    # Використовуємо Bootstrap, що забезпечує більшість стилів
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

# ІМПОРТИ КОЛБЕКІВ
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


# Функція для створення елемента навігації у стилі кнопки
def nav_button(text, href, target="_self"):
    return dbc.NavItem(
        dcc.Link(
            text,
            href=href,
            className="nav-link nav-button-style",
            target=target,
        ),
        className="mx-1",
    )


navbar = dbc.Navbar(
    dbc.Container(
        [
            # *** ВИПРАВЛЕНО ***
            # Видалено target="_blank", щоб уникнути TypeError.
            # Посилання на зовнішній звіт залишилося.
            dbc.NavbarBrand(
                "GeoHydroAI",
                href="https://www.geohydroai.org/reports/accuracy_dem_story.html",
                class_name="me-auto"
            ),

            # Кнопка-гамбургер для мобільних пристроїв
            dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),

            # Контент, що згортається (навигация)
            dbc.Collapse(
                dbc.Nav(
                    [
                        nav_button("Dashboard", url("/dashboard")),
                        nav_button("DEM Difference", url("/dem-diff")),
                        nav_button("Flood Scenarios (Map)", url("/flood-dem-diif")),
                        nav_button("FFA Report", "https://geohydroai.org/reports/ffa_report_en.html", target="_blank"),
                        nav_button("Cross Section", "https://geohydroai.org/reports/cross_section_dashboard.html",
                                   target="_blank"),
                    ],
                    className="justify-content-end", # Вирівнювання праворуч
                    navbar=True,
                ),
                id="navbar-collapse",
                is_open=False,
                navbar=True,
            ),
        ],
        fluid=True,
    ),
    color="dark",
    dark=True,
    sticky="top",
    class_name="py-1 gh-navbar",
    expand="lg",
)


# Колбек для роботи кнопки-гамбургера (toggler)
@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


app.layout = html.Div([dcc.Location(id="url"), navbar, dash.page_container])

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)
