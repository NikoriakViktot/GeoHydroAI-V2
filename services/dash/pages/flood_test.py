# pages/flood_test.py
import os
import json
import logging
from typing import Dict, List, Tuple, Set

import geopandas as gpd
import dash
from dash import html, dcc, callback, Output, Input, no_update
import dash_leaflet as dl

from registry import get_df

# ---------- Page & logging ----------
dash.register_page(__name__, path="/flood-test", name="Flood Scenarios Test", order=99)
app = dash.get_app()

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=os.getenv("PAGE_LOG_LEVEL", "INFO"),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

TC_BASE = os.getenv("TERRACOTTA_PUBLIC_URL", "https://www.geohydroai.org/tc").rstrip("/")
ASSETS_INDEX_PATH = "assets/layers_index.json"

# ---------- Basin (optional overlay) ----------
try:
    basin: gpd.GeoDataFrame = get_df("basin")
    basin = basin.to_crs("EPSG:4326")
    BASIN_JSON = json.loads(basin.to_json())
    logger.info("Basin loaded: CRS=%s, rows=%d", basin.crs, len(basin))
except Exception as e:
    logger.warning("Basin not available: %s", e)
    BASIN_JSON = None

# ---------- Helpers ----------
def _fix_path(p: str) -> str:
    if not p:
        return p
    p = p.replace("\\", "/")
    if p.startswith("/"):
        return os.path.normpath(p)
    if p.startswith("data/COG/"):
        return "/app/data/cogs/" + p.split("data/COG/")[1]
    if p.startswith("data/cogs/"):
        return "/app/data/cogs/" + p.split("data/cogs/")[1]
    if p.startswith("data/"):
        return "/app/" + p
    return os.path.normpath(p)

def _parse_level(s: str) -> int:
    """'1m' -> 1, '10m' -> 10; fallback 0"""
    try:
        return int(str(s).lower().replace("m", "").strip())
    except Exception:
        return 0

def build_dem_url(dem_name: str, cmap: str, stretch) -> str:
    s = f"[{stretch[0]},{stretch[1]}]"
    return f"{TC_BASE}/singleband/dem/{dem_name}" + "/{z}/{x}/{y}.png" + f"?colormap={cmap}&stretch_range={s}"

def build_flood_url(dem_name: str, hand_name: str, level: str, cmap: str, stretch, pure_blue: bool) -> str:
    """
    Terracotta layer name pattern:
        {dem_name}_{hand_name}_flood_{level}
    where hand_name like 'hand_2000', level like '1m'.
    """
    layer = f"{dem_name}_{hand_name}_flood_{level}"
    s = f"[{stretch[0]},{stretch[1]}]"
    base = f"{TC_BASE}/singleband/flood_scenarios/{layer}" + "/{z}/{x}/{y}.png"
    return f"{base}?colormap=custom&colors=0000ff&stretch_range={s}" if pure_blue \
           else f"{base}?colormap={cmap}&stretch_range={s}"

# ---------- Read & normalize layers_index.json ----------
# Fallback DEM list if index missing/broken
DEM_LIST: List[str] = ["alos_dem", "aster_dem", "copernicus_dem", "fab_dem", "nasa_dem", "srtm_dem", "tan_dem"]

layers_index: List[dict] = []
try:
    with open(ASSETS_INDEX_PATH, "r") as f:
        raw_index = json.load(f)
    items = raw_index if isinstance(raw_index, list) else [raw_index]
    for rec in items:
        r = dict(rec)
        if r.get("path"):
            r["path"] = _fix_path(r["path"])
        layers_index.append(r)
    logger.info("Layers index normalized, entries=%d", len(layers_index))
except Exception as e:
    logger.warning("Failed to read layers index '%s': %s", ASSETS_INDEX_PATH, e)
    layers_index = []

# Build maps from normalized index:
#   DEM_TO_HANDS[dem] -> ["hand_2000", ...]
#   DEM_TO_LEVELS[(dem, hand)] -> ["1m","2m",...]
DEM_TO_HANDS: Dict[str, List[str]] = {}
DEM_TO_LEVELS: Dict[Tuple[str, str], List[str]] = {}

if layers_index:
    for r in layers_index:
        if r.get("category") != "flood_scenarios":
            continue
        dem = r.get("dem")
        hand = r.get("hand")      # e.g. 'hand_2000'
        level = r.get("flood")    # e.g. '1m'
        if dem and hand and level:
            DEM_TO_HANDS.setdefault(dem, set()).add(hand)
            DEM_TO_LEVELS.setdefault((dem, hand), set()).add(level)

    if DEM_TO_HANDS:
        for dem, hands_set in DEM_TO_HANDS.items():
            DEM_TO_HANDS[dem] = sorted(hands_set)
        for key, levels_set in DEM_TO_LEVELS.items():
            DEM_TO_LEVELS[key] = sorted(levels_set, key=_parse_level)
        DEM_LIST = sorted(DEM_TO_HANDS.keys())

logger.info("DEM list: %s", ", ".join(DEM_LIST))
logger.info("DEM_TO_HANDS keys: %s", ", ".join(DEM_TO_HANDS.keys()))

# ---------- UI constants ----------
COLORMAPS = ["viridis", "terrain"]
BASE_KEYS = ["toner", "terrain", "osm"]
URL_TEMPLATE = {
       "osm":     "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
}
ATTRIBUTION = (
    ' <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
)

# ---------- Layout ----------
layout = html.Div([
    html.H4("Flood Scenario Test Map"),
    html.Div([
        html.Div([
            html.Label("DEM"),
            dcc.Dropdown(
                id="dem-name",
                options=[{"label": d, "value": d} for d in DEM_LIST],
                value=DEM_LIST[0],
                style={"width": 220},
                clearable=False
            ),
        ], style={"display": "inline-block", "marginRight": 16}),

        html.Div([
            html.Label("HAND"),
            dcc.Dropdown(id="hand-name", options=[], placeholder="hand_*", clearable=False),
        ], style={"display": "inline-block", "width": 160, "marginRight": 16}),

        html.Div([
            html.Label("Flood level (m)"),
            dcc.Dropdown(id="flood-level", options=[], placeholder="1–10 m", clearable=False),
        ], style={"display": "inline-block", "width": 160, "marginRight": 16}),

        html.Div([
            html.Label("DEM Colormap"),
            dcc.Dropdown(
                id="dem-cmap",
                options=[{"label": c.capitalize(), "value": c} for c in COLORMAPS],
                value="viridis",
                style={"width": 160},
                clearable=False
            ),
        ], style={"display": "inline-block", "marginRight": 16}),

        html.Div([
            html.Label("DEM Stretch"),
            dcc.RangeSlider(
                id="dem-stretch",
                min=0, max=4000, step=50, value=[250, 2200],
                marks={0:"0",1000:"1000",2000:"2000",3000:"3000",4000:"4000"},
                tooltip={"always_visible": False, "placement": "top"},
            ),
        ], style={"display": "inline-block", "width": 360, "marginRight": 16}),

        html.Div([
            html.Label("Flood Colormap"),
            dcc.Dropdown(
                id="flood-cmap",
                options=[
                    {"label": "Blues", "value": "blues"},
                    {"label": "Viridis", "value": "viridis"},
                    {"label": "Pure Blue", "value": "custom"},
                ],
                value="blues",
                style={"width": 160},
                clearable=False
            ),
        ], style={"display": "inline-block", "marginRight": 16}),

        html.Div([
            html.Label("Flood Stretch"),
            dcc.RangeSlider(
                id="flood-stretch",
                min=0, max=10, step=1, value=[0, 5],
                marks={i: str(i) for i in range(0, 11)},
                tooltip={"always_visible": False, "placement": "top"}
            ),
        ], style={"display": "inline-block", "width": 260}),
    ], style={"marginBottom": 14}),

    dl.Map([
        dl.LayersControl([
            *[
                dl.BaseLayer(
                    dl.TileLayer(url=URL_TEMPLATE[k], attribution=ATTRIBUTION),
                    name=k.capitalize(), checked=(k == "toner")
                ) for k in BASE_KEYS
            ],
            dl.Overlay(dl.TileLayer(id="dem-tiles",   url="", opacity=0.75), name="DEM",   checked=True),
            dl.Overlay(dl.TileLayer(id="flood-tiles", url="", opacity=1.0),  name="Flood", checked=True),
            *([
                dl.Overlay(
                    dl.GeoJSON(
                        data=BASIN_JSON, id="basin-flood",
                        options={"style": {"color": "blue", "weight": 2, "fill": False}}
                    ),
                    name="Basin", checked=True
                )
            ] if BASIN_JSON else [])
        ], id="lc", position="topright"),
    ], style={"width": "100%", "height": "700px"}, center=[47.8, 25.03], zoom=10),

    html.Div(id="log"),
])

# ---------- Callbacks ----------
@callback(
    Output("hand-name", "options"),
    Output("hand-name", "value"),
    Input("dem-name", "value"),
)
def update_hands_for_dem(dem_name: str):
    hands = DEM_TO_HANDS.get(dem_name) or []
    options = [{"label": h, "value": h} for h in hands]
    default = hands[0] if hands else None
    return options, default

@callback(
    Output("flood-level", "options"),
    Output("flood-level", "value"),
    Input("dem-name", "value"),
    Input("hand-name", "value"),
)
def update_levels_for_dem_hand(dem_name: str, hand_name: str):
    levels = DEM_TO_LEVELS.get((dem_name, hand_name)) or []
    options = [{"label": f"{_parse_level(l)} m", "value": l} for l in levels]
    default = "5m" if "5m" in levels else (levels[0] if levels else None)
    return options, default

@callback(
    Output("dem-tiles", "url"),
    Output("flood-tiles", "url"),
    Input("dem-name", "value"),
    Input("dem-cmap", "value"),
    Input("dem-stretch", "value"),
    Input("hand-name", "value"),
    Input("flood-level", "value"),
    Input("flood-cmap", "value"),
    Input("flood-stretch", "value"),
)
def update_tc_urls(dem_name, dem_cmap, dem_stretch, hand_name, flood_level, flood_cmap, flood_stretch):
    if not dem_name:
        return no_update, no_update

    dem_url = build_dem_url(dem_name, dem_cmap or "viridis", dem_stretch or [250, 2200])

    levels = DEM_TO_LEVELS.get((dem_name, hand_name)) or []
    if not (hand_name and flood_level and flood_level in levels):
        flood_url = ""
    else:
        pure_blue = (flood_cmap == "custom")
        flood_url = build_flood_url(
            dem_name, hand_name, flood_level,
            flood_cmap or "blues",
            flood_stretch or [0, 5],
            pure_blue=pure_blue
        )

    logger.debug("DEM URL: %s", dem_url)
    logger.debug("FLOOD URL: %s", flood_url)
    return dem_url, flood_url

@callback(
    Output("log", "children"),
    Input("lc", "baseLayer"),
    Input("lc", "overlays"),
    prevent_initial_call=True
)
def log_layers(base_layer, overlays):
    return f"Base layer: {base_layer}, overlays: {json.dumps(overlays)}"



# import dash
# from dash import html, dcc, callback, Output, Input
# import dash_leaflet as dl
# import json
# import geopandas as gpd
# from registry import get_df
#
# print("Loaded flood test page")
#
# dash.register_page(__name__, path="/flood-test", name="Flood Scenarios Test", order=99)
# app = dash.get_app()
#
# # Завантаження шару басейну
# try:
#     # basin = gpd.read_file("data/basin_bil_cher_4326.gpkg")
#     basin: gpd.GeoDataFrame = get_df("basin")
#     print("Basin loaded! CRS:", basin.crs)
#     basin = basin.to_crs("EPSG:4326")
#     basin_json = json.loads(basin.to_json())
# except Exception as e:
#     print("❌ Error loading basin:", e)
#     basin_json = None
#
# # Flood scenario options
# flood_options = [
#     {"label": "Flood 1m", "value": "alos_dem_hand_2000_flood_1m"},
#     {"label": "Flood 2m", "value": "alos_dem_hand_2000_flood_2m"},
#     {"label": "Flood 3m", "value": "alos_dem_hand_2000_flood_3m"},
#     {"label": "Flood 5m", "value": "alos_dem_hand_2000_flood_5m"},
#     {"label": "Flood 10m", "value": "alos_dem_hand_2000_flood_10m"},
# ]
#
# colormaps = ["viridis", "terrain"]
#
# base_keys = ["toner", "terrain", "osm"]
# url_template = {
#     "toner": "http://{{s}}.tile.stamen.com/toner/{{z}}/{{x}}/{{y}}.png",
#     "terrain": "http://{{s}}.tile.stamen.com/terrain/{{z}}/{{x}}/{{y}}.png",
#     "osm": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
# }
# attribution = (
#     'Map tiles by <a href="http://stamen.com">Stamen Design</a>, '
#     '<a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a> &mdash; Map data '
#     '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
# )
#
# layout = html.Div([
#     html.H4("Flood Scenario Test Map"),
#     html.Div([
#         html.Div([
#             html.Label("DEM Colormap:"),
#             dcc.Dropdown(
#                 id="colormap-dropdown-flood",
#                 options=[{"label": c.capitalize(), "value": c} for c in colormaps],
#                 value="viridis",
#                 style={"width": 180}
#             ),
#         ], style={"display": "inline-block", "marginRight": 20}),
#         html.Div([
#             html.Label("Flood Scenario:"),
#             dcc.Dropdown(
#                 id="flood-scenario-dropdown-flood",
#                 options=flood_options,
#                 value="alos_dem_hand_2000_flood_5m",
#                 placeholder="Select flood scenario",
#                 style={"width": 200}
#             ),
#         ], style={"display": "inline-block", "marginRight": 20}),
#         html.Div([
#             html.Label("Flood Colormap:"),
#             dcc.Dropdown(
#                 id="flood-colormap-dropdown-flood",
#                 options=[
#                     {"label": "Blues", "value": "blues"},
#                     {"label": "Viridis", "value": "viridis"},
#                     {"label": "Pure Blue", "value": "custom"}
#                 ],
#                 value="blues",
#                 style={"width": 150}
#             ),
#         ], style={"display": "inline-block", "marginRight": 20}),
#         html.Div([
#             html.Label("Flood Stretch Range:"),
#             dcc.RangeSlider(
#                 id="flood-stretch-slider",
#                 min=0, max=10, step=1, value=[0, 5],
#                 marks={i: str(i) for i in range(0, 11)},
#                 tooltip={"always_visible": False, "placement": "top"}
#             ),
#         ], style={"width": 200, "display": "inline-block"}),
#     ], style={"marginBottom": 18}),
#     dl.Map([
#         dl.LayersControl([
#             *[
#                 dl.BaseLayer(
#                     dl.TileLayer(url=url_template[key], attribution=attribution),
#                     name=key.capitalize(),
#                     checked=(key == "toner")
#                 ) for key in base_keys
#             ],
#             dl.Overlay(
#                 dl.TileLayer(
#                     id="dem-tile-flood",
#                     url="/tc/singleband/dem/fab_dem/{z}/{x}/{y}.png?colormap=viridis&stretch_range=[0,2200]",
#                     opacity=0.7
#                 ),
#                 name="DEM",
#                 checked=True
#             ),
#             dl.Overlay(
#                 dl.TileLayer(
#                     id="flood-tile-flood",
#                     url="",
#                     opacity=1.0
#                 ),
#                 name="Flood",
#                 checked=True
#             ),
#             *([
#                   dl.Overlay(
#                       dl.GeoJSON(
#                           data=basin_json,
#                           id="basin-flood",
#                           options={"style": {"color": "blue", "weight": 2, "fill": False}}
#                       ),
#                       name="Basin",
#                       checked=True
#                   )
#               ] if basin_json else []),
#         ], id="lc", position="topright"),
#     ], style={'width': '100%', 'height': '700px'}, center=[47.8, 25.03], zoom=10),
#     html.Div(id="log"),  # <-- Окремо від Map!
# ])
# @app.callback(
#     Output("dem-tile-flood", "url"),
#     Output("flood-tile-flood", "url"),
#     Input("colormap-dropdown-flood", "value"),
#     Input("flood-scenario-dropdown-flood", "value"),
#     Input("flood-colormap-dropdown-flood", "value"),
#     Input("flood-stretch-slider", "value"),
# )
# def update_tile_urls(dem_colormap, flood_name, flood_colormap, flood_stretch):
#     print(f"update_tile_urls: DEM={dem_colormap}, Flood={flood_name}, FloodMap={flood_colormap}, Stretch={flood_stretch}")
#     stretch = "[250,2200]"
#     dem_url = f"/tc/singleband/dem/fab_dem/{{z}}/{{x}}/{{y}}.png?colormap={dem_colormap}&stretch_range={stretch}"
#     if flood_name:
#         flood_stretch_str = f"[{flood_stretch[0]},{flood_stretch[1]}]"
#         if flood_colormap == "custom":
#             flood_url = (
#                 f"/tc/singleband/flood_scenarios/{flood_name}/{{z}}/{{x}}/{{y}}.png"
#                 f"?colormap=custom&colors=0000ff&stretch_range={flood_stretch_str}"
#             )
#         else:
#             flood_url = (
#                 f"/tc/singleband/flood_scenarios/{flood_name}/{{z}}/{{x}}/{{y}}.png"
#                 f"?colormap={flood_colormap}&stretch_range={flood_stretch_str}"
#             )
#     else:
#         flood_url = ""
#     print("DEM URL:", dem_url)
#     print("Flood URL:", flood_url)
#     return dem_url, flood_url
#
# # Callback for logging current selection
# @app.callback(
#     Output("log", "children"),
#     Input("lc", "baseLayer"),
#     Input("lc", "overlays"),
#     prevent_initial_call=True
# )
# def log_layers(base_layer, overlays):
#     return f"Base layer: {base_layer}, overlays: {json.dumps(overlays)}"
