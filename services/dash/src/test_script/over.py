import dash
from dash import html
import dash_leaflet as dl

app = dash.Dash(__name__)

app.layout = html.Div([
    dl.Map([
        dl.LayersControl([
            dl.BaseLayer(
                dl.TileLayer(url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"),
                name="OSM", checked=True
            ),
            dl.Overlay(
                dl.Marker(position=[56, 10]),
                name="Marker", checked=True
            )
        ], id="lc")
    ], style={'height': '70vh'}, center=[56, 10], zoom=7),
])

if __name__ == "__main__":
    app.run(debug=True)
