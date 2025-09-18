# pages/test_map.py
import dash
from dash import html, dcc, callback, Input, Output
import dash_leaflet as dl
import json
import geopandas as gpd
from registry import get_df

dash.register_page(__name__, path="/test", name="Dashboard", order=1)

def _load_basin_geojson() -> dict | None:
    try:
        basin: gpd.GeoDataFrame = get_df("basin")
        basin = basin.to_crs("EPSG:4326")
        return json.loads(basin.to_json())
    except Exception as e:
        print("❌ Error loading basin:", e)
        return None

def layout():
    basin_json = _load_basin_geojson()
    return html.Div([
        dcc.Dropdown(
            id="colormap-dropdown-test",
            options=[{"label": c, "value": c} for c in ["viridis","terrain","inferno","jet","spectral","rainbow"]],
            value="viridis"
        ),
        dl.Map([
            dl.TileLayer(),
            (dl.GeoJSON(
                data=basin_json,
                id="basin-test",
                options={"style": {"color": "blue", "weight": 2, "fill": False}},
            ) if basin_json is not None else html.Div("❌ Basin not loaded!")),
            dl.TileLayer(
                id="dem-tile-test",
                url="/tc/singleband/dem/fab_dem/{z}/{x}/{y}.png?colormap=viridis&stretch_range=[0,2200]",
                opacity=0.7,
            ),
        ], style={'width': '100%', 'height': '700px'}, center=[47.8, 25.03], zoom=10),
    ])

@callback(
    Output("dem-tile-test", "url"),
    Input("colormap-dropdown-test", "value"),
)
def update_tile_url(colormap: str):
    stretch = "[0,2200]"
    return f"/tc/singleband/dem/fab_dem/{{z}}/{{x}}/{{y}}.png?colormap={colormap}&stretch_range={stretch}"
