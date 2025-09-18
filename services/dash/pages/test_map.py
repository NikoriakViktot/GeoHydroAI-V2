import dash
from dash import html, dcc, callback
import dash_leaflet as dl
import json
import geopandas as gpd
from dash.dependencies import Input, Output
from config import gdf_basin


print("Loaded test page")  # показує, що файл точно імпортується

dash.register_page(__name__, path="/test", name="Dashboard", order=1)

# Читання шару басейну (GeoJSON)
try:
    # basin = gpd.read_file("data/basin_bil_cher_4326.gpkg")
    basin = gdf_basin
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
            url="/tc/singleband/dem/fab_dem/{z}/{x}/{y}.png?colormap=viridis&stretch_range=[0,2200]",  # Дефолтний URL
            opacity=0.7
        )
    ], style={'width': '100%', 'height': '700px'}, center=[47.8, 25.03], zoom=10)
])

@callback(
    Output("dem-tile-test", "url"),
    Input("colormap-dropdown-test", "value")
)
def update_tile_url(colormap):
    try:
        print("Callback called! colormap:", colormap)
        stretch = "[0,2200]"
        url = f"/tc/singleband/dem/fab_dem/{{z}}/{{x}}/{{y}}.png?colormap={colormap}&stretch_range={stretch}"
        print("Returning url:", url)
        return url
    except Exception as e:
        import traceback
        print("EXCEPTION in update_tile_url:", e)
        traceback.print_exc()
        return dash.no_update
