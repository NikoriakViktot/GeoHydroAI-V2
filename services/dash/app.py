# app.py  --- BASELINE RESTORED

import dash
from dash import html, dcc
from werkzeug.middleware.dispatcher import DispatcherMiddleware

from init_tc_server import tile_server  # Terracotta tile-server

external_stylesheets = [
    "https://cdn.jsdelivr.net/npm/bootswatch@5.1.3/dist/darkly/bootstrap.min.css"
]

app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=external_stylesheets,
    suppress_callback_exceptions=True,
    update_title=None,   # не миготіти "Updating..."
)

app.title = "GeoHydroAI | DEM OLAP"
server = app.server
application = DispatcherMiddleware(server, {"/tc": tile_server})

# --- Статичний navbar (як ти мав спочатку) ----------------------------------
navbar = html.Div(
    [
        dcc.Link("Dashboard", href="/", style={"color": "#00bfff", "marginRight": "24px"}),
        dcc.Link("DEM Diff Analysis", href="/dem-diff", style={"color": "#00bfff", "marginRight": "24px"}),
        dcc.Link("TestMap",  href="/test",    style={"color": "#00bfff", "marginRight": "24px"}),
        dcc.Link("Flood Scenarios Test", href="/flood-test", style={"color": "#00bfff", "marginRight": "24px"}),

    ],

    style={"padding": "20px 0"}
)

# --- Головний layout: navbar + page_container -------------------------------
#     Контент тягнеться на всю ширину (мінус лівий sidebar 300px).
app.layout = html.Div([
    html.Div([
        navbar,
        dash.page_container,
    ],
    style={
        "marginLeft": "300px",          # резерв під фіксований сайдбар сторінок
        "padding": "0 30px 24px 30px",
        "backgroundColor": "#181818",
        "minHeight": "100vh",
        "width": "calc(100% - 300px)",  # страхує на мобільних; можна прибрати
    })
])

print("PAGES:", list(dash.page_registry.keys()))

if __name__ == "__main__":
    from werkzeug.serving import run_simple
    run_simple(
        "0.0.0.0",
        8050,
        application,
        use_reloader=True,
        use_debugger=True,
        use_evalex=True,
        passthrough_errors=True
    )
