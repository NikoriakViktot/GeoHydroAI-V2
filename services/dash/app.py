# app.py
import dash
from dash import html, dcc

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

navbar = html.Div(
    [
        dcc.Link("Dashboard", href="/", style={"color": "#00bfff", "marginRight": "24px"}),
        dcc.Link("DEM Diff Analysis", href="/dem-diff", style={"color": "#00bfff", "marginRight": "24px"}),
        dcc.Link("ICESat-2 Map", href="/icesat-map", style={"color": "#00bfff", "marginRight": "24px"}),
        dcc.Link("Best DEM", href="/best-dem", style={"color": "#00bfff", "marginRight": "24px"}),
        dcc.Link("CDF Accumulation", href="/cdf", style={"color": "#00bfff", "marginRight": "24px"}),
        dcc.Link("Flood Scenarios Test", href="/flood-test", style={"color": "#00bfff", "marginRight": "24px"}),
    ],
    style={"padding": "20px 0"},
)

app.layout = html.Div([navbar, dash.page_container])

# ВАЖЛИВО: просто імпортуємо модуль з колбеками — вони зареєструються.
import callbacks.main_callbacks  # noqa: F401

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)
