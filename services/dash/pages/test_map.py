# pages/test_map.py
import dash
from dash import html, dcc, callback, Input, Output
import dash_leaflet as dl
import json
import geopandas as gpd
from registry import get_df

print("Loaded test page")

dash.register_page(__name__, path="/test", name="Dashboard", order=1)

# Читання шару басейну (через registry)
try:
    basin: gpd.GeoDataFrame = get_df("basin")
    print("Basin loaded! CRS:", basin.crs)
    basin = basin.to_crs("EPSG:4326")
    basin_json = json.loads(basin.to_json())
except Exception as e:
    print("❌ Error loading basin:", e)
    basin_json = None

colormaps = ["viridis", "terrain", "inferno", "jet", "spectral", "rainbow"]

layout = html.Div([
    dcc.Dropdown(
        id="colormap-dropdown-test",
        options=[{"label": c, "value": c} for c in colormaps],
        value="viridis"
    ),
    dl.Map([
        dl.TileLayer(),
        dl.GeoJSON(
            data=basin_json,
            id="basin-test",
            options={
                "style": {
                    "color": "blue",
                    "weight": 2,
                    "fill": False,
                }
            }
        ) if basin_json is not None else html.Div("❌ Basin not loaded!"),
        dl.TileLayer(
            id="dem-tile-test",
            url="/tc/singleband/dem/fab_dem/{z}/{x}/{y}.png?colormap=viridis&stretch_range=[0,2200]",
            opacity=0.7
        )
    ], style={'width': '100%', 'height': '700px'}, center=[47.8, 25.03], zoom=10)
])

@callback(
    Output("dem-tile-test", "url"),
    Input("colormap-dropdown-test", "value")
)
def update_tile_url(colormap: str):
    try:
        stretch = "[0,2200]"
        url = f"/tc/singleband/dem/fab_dem/{{z}}/{{x}}/{{y}}.png?colormap={colormap}&stretch_range={stretch}"
        return url
    except Exception:
        return dash.no_update
