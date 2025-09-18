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
server = app.server  # важливо для gunicorn: app:server

navbar = html.Div(
    [
        dcc.Link("Dashboard", href="/", style={"color": "#00bfff", "marginRight": "24px"}),
        dcc.Link("DEM Diff Analysis", href="/dem-diff", style={"color": "#00bfff", "marginRight": "24px"}),
        dcc.Link("TestMap", href="/test", style={"color": "#00bfff", "marginRight": "24px"}),
        dcc.Link("Flood Scenarios Test", href="/flood-test", style={"color": "#00bfff", "marginRight": "24px"}),
    ],
    style={"padding": "20px 0"},
)

app.layout = html.Div([navbar, dash.page_container])

if __name__ == "__main__":
    # локальний запуск без gunicorn
    app.run_server(host="0.0.0.0", port=8050, debug=True)
