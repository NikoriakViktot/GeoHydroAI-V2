#layout/tracks_profile_tab.py


from dash import html
import dash_leaflet as dl
import json
import geopandas as gpd

from dash import html, dcc
from utils.style import dark_card_style, dropdown_style
from registry import get_df

YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

try:
    basin: gpd.GeoDataFrame = get_df("basin")
    print("Basin loaded! CRS:", basin.crs)
    basin = basin.to_crs("EPSG:4326")
    basin_json = json.loads(basin.to_json())
except Exception as e:
    print("❌ Error loading basin:", e)
    basin_json = None



tracks_map_layout = html.Div([
    html.H4("Карта ICESat-2 треків", style={"color": "#EEEEEE"}),
    html.Div([
        dcc.Dropdown(
            id="year_dropdown",
            options=[{"label": str(y), "value": y} for y in YEARS],
            value=YEARS[-1], clearable=False,
            style={**dropdown_style, "width": "110px", "display": "inline-block"}
        ),
        dcc.Dropdown(
            id="track_rgt_spot_dropdown",
            options=[],  # генерується callback-ом по року!
            style={**dropdown_style, "width": "300px", "display": "inline-block", "marginLeft": "8px"}
        ),
        dcc.Dropdown(
            id="date_dropdown",
            options=[],  # генерується callback-ом по треку!
            style={**dropdown_style, "width": "150px", "display": "inline-block", "marginLeft": "8px"}
        ),
    ], style={"marginBottom": "12px"}),

    dl.Map([
        dl.TileLayer(
            url="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
            attribution="© OpenTopoMap contributors"
        ),
        dl.LayerGroup(id="point_group"),
        dl.GeoJSON(data=basin, id="basin_track_map", options={"style": {"color": "blue", "weight": 2, "fill": False}})
    ],
        style={
            "height": "520px",
            "width": "100%",
            "minWidth": "350px",
            "marginTop": "10px",
        },
        center=[47.8, 25.03],
        zoom=10,
        id="leaflet_map"
    )



], style={
    "backgroundColor": "#181818",
    "color": "#EEEEEE",
    "padding": "18px 12px 32px 12px",
    "minHeight": "480px",
})

# html.Div([
#     html.Button(
#         "← Повернутися до графіку",
#         id="go_to_profile_btn",
#         n_clicks=0,
#         style={
#             "marginTop": "32px",
#             "fontSize": "17px",
#             "background": "#253152",
#             "color": "#B9E0FF",
#             "border": "none",
#             "borderRadius": "12px",
#             "padding": "12px 28px",
#             "fontWeight": "bold",
#             "cursor": "pointer",
#             "boxShadow": "0 2px 8px #10161a33"
#         }
#     ),
# ], style={"display": "flex", "justifyContent": "center"}),