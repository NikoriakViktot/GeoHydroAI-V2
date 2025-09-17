import dash
from dash import html, dcc, callback, Output, Input
import dash_leaflet as dl
import json
import geopandas as gpd

print("Loaded flood test page")

dash.register_page(__name__, path="/flood-test", name="Flood Scenarios Test", order=99)

# Завантаження шару басейну
try:
    basin = gpd.read_file("data/basin_bil_cher_4326.gpkg")
    print("Basin loaded! CRS:", basin.crs)
    basin = basin.to_crs("EPSG:4326")
    basin_json = json.loads(basin.to_json())
except Exception as e:
    print("❌ Error loading basin:", e)
    basin_json = None

# Flood scenario options
flood_options = [
    {"label": "Flood 1m", "value": "alos_dem_hand_2000_flood_1m"},
    {"label": "Flood 2m", "value": "alos_dem_hand_2000_flood_2m"},
    {"label": "Flood 3m", "value": "alos_dem_hand_2000_flood_3m"},
    {"label": "Flood 5m", "value": "alos_dem_hand_2000_flood_5m"},
    {"label": "Flood 10m", "value": "alos_dem_hand_2000_flood_10m"},
]

colormaps = ["viridis", "terrain"]

base_keys = ["toner", "terrain", "osm"]
url_template = {
    "toner": "http://{{s}}.tile.stamen.com/toner/{{z}}/{{x}}/{{y}}.png",
    "terrain": "http://{{s}}.tile.stamen.com/terrain/{{z}}/{{x}}/{{y}}.png",
    "osm": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
}
attribution = (
    'Map tiles by <a href="http://stamen.com">Stamen Design</a>, '
    '<a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a> &mdash; Map data '
    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
)

layout = html.Div([
    html.H4("Flood Scenario Test Map"),
    html.Div([
        html.Div([
            html.Label("DEM Colormap:"),
            dcc.Dropdown(
                id="colormap-dropdown-flood",
                options=[{"label": c.capitalize(), "value": c} for c in colormaps],
                value="viridis",
                style={"width": 180}
            ),
        ], style={"display": "inline-block", "marginRight": 20}),
        html.Div([
            html.Label("Flood Scenario:"),
            dcc.Dropdown(
                id="flood-scenario-dropdown-flood",
                options=flood_options,
                value="alos_dem_hand_2000_flood_5m",
                placeholder="Select flood scenario",
                style={"width": 200}
            ),
        ], style={"display": "inline-block", "marginRight": 20}),
        html.Div([
            html.Label("Flood Colormap:"),
            dcc.Dropdown(
                id="flood-colormap-dropdown-flood",
                options=[
                    {"label": "Blues", "value": "blues"},
                    {"label": "Viridis", "value": "viridis"},
                    {"label": "Pure Blue", "value": "custom"}
                ],
                value="blues",
                style={"width": 150}
            ),
        ], style={"display": "inline-block", "marginRight": 20}),
        html.Div([
            html.Label("Flood Stretch Range:"),
            dcc.RangeSlider(
                id="flood-stretch-slider",
                min=0, max=10, step=1, value=[0, 5],
                marks={i: str(i) for i in range(0, 11)},
                tooltip={"always_visible": False, "placement": "top"}
            ),
        ], style={"width": 200, "display": "inline-block"}),
    ], style={"marginBottom": 18}),
    dl.Map([
        dl.LayersControl([
            *[
                dl.BaseLayer(
                    dl.TileLayer(url=url_template[key], attribution=attribution),
                    name=key.capitalize(),
                    checked=(key == "toner")
                ) for key in base_keys
            ],
            dl.Overlay(
                dl.TileLayer(
                    id="dem-tile-flood",
                    url="/tc/singleband/dem/fab_dem/{z}/{x}/{y}.png?colormap=viridis&stretch_range=[0,2200]",
                    opacity=0.7
                ),
                name="DEM",
                checked=True
            ),
            dl.Overlay(
                dl.TileLayer(
                    id="flood-tile-flood",
                    url="",
                    opacity=1.0
                ),
                name="Flood",
                checked=True
            ),
            *([
                  dl.Overlay(
                      dl.GeoJSON(
                          data=basin_json,
                          id="basin-flood",
                          options={"style": {"color": "blue", "weight": 2, "fill": False}}
                      ),
                      name="Basin",
                      checked=True
                  )
              ] if basin_json else []),
        ], id="lc", position="topright"),
    ], style={'width': '100%', 'height': '700px'}, center=[47.8, 25.03], zoom=10),
    html.Div(id="log"),  # <-- Окремо від Map!
])
@callback(
    Output("dem-tile-flood", "url"),
    Output("flood-tile-flood", "url"),
    Input("colormap-dropdown-flood", "value"),
    Input("flood-scenario-dropdown-flood", "value"),
    Input("flood-colormap-dropdown-flood", "value"),
    Input("flood-stretch-slider", "value"),
)
def update_tile_urls(dem_colormap, flood_name, flood_colormap, flood_stretch):
    print(f"update_tile_urls: DEM={dem_colormap}, Flood={flood_name}, FloodMap={flood_colormap}, Stretch={flood_stretch}")
    stretch = "[250,2200]"
    dem_url = f"/tc/singleband/dem/fab_dem/{{z}}/{{x}}/{{y}}.png?colormap={dem_colormap}&stretch_range={stretch}"
    if flood_name:
        flood_stretch_str = f"[{flood_stretch[0]},{flood_stretch[1]}]"
        if flood_colormap == "custom":
            flood_url = (
                f"/tc/singleband/flood_scenarios/{flood_name}/{{z}}/{{x}}/{{y}}.png"
                f"?colormap=custom&colors=0000ff&stretch_range={flood_stretch_str}"
            )
        else:
            flood_url = (
                f"/tc/singleband/flood_scenarios/{flood_name}/{{z}}/{{x}}/{{y}}.png"
                f"?colormap={flood_colormap}&stretch_range={flood_stretch_str}"
            )
    else:
        flood_url = ""
    print("DEM URL:", dem_url)
    print("Flood URL:", flood_url)
    return dem_url, flood_url

# Callback for logging current selection
@callback(
    Output("log", "children"),
    Input("lc", "baseLayer"),
    Input("lc", "overlays"),
    prevent_initial_call=True
)
def log_layers(base_layer, overlays):
    return f"Base layer: {base_layer}, overlays: {json.dumps(overlays)}"
