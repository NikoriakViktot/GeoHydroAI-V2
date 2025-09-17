import dash
from dash import html, dcc
external_stylesheets = [
    "https://cdn.jsdelivr.net/npm/bootswatch@5.1.3/dist/darkly/bootstrap.min.css"
]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, use_pages=True)

app.layout = html.Div([
    html.H2("GeoHydro Dashboard"),
    dcc.Link("Dashboard", href="/"),
    " | ",
    dcc.Link("Карти ICESat-2", href="/tracks-map"),
    dash.page_container
])

if __name__ == "__main__":
    app.run(debug=True)


