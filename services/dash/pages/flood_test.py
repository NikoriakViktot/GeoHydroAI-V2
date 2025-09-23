# pages/flood_map_deck.py
import os, json, logging, sys
from typing import Dict, List, Tuple

import geopandas as gpd
import dash
from dash import html, dcc, callback, Output, Input, no_update
import dash_deckgl

from registry import get_df

# ---------- Page ----------
dash.register_page(__name__, path="/flood-map", name="Flood Scenarios", order=98)
app = dash.get_app()

# ---------- Logging to stdout ----------
logger = logging.getLogger("pages.flood_map_deck")
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    h = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
logger.setLevel(getattr(logging, os.getenv("PAGE_LOG_LEVEL", "INFO").upper(), logging.INFO))
logger.propagate = False
from urllib.parse import urlparse, urlunparse

def _strip_www(u: str) -> str:
    p = urlparse(u)
    host = p.netloc.replace("www.", "")
    return urlunparse((p.scheme or "https", host, p.path, "", "", "")).rstrip("/")


TC_BASE = _strip_www(os.getenv("TERRACOTTA_PUBLIC_URL", "https://geohydroai.org/tc"))
MAPBOX_ACCESS_TOKEN = os.getenv("MAPBOX_ACCESS_TOKEN", "").strip()
ASSETS_INDEX_PATH = "assets/layers_index.json"
# ---------- Read & normalize layers_index.json ----------
DEM_LABELS = {
    "tan_dem": "TanDEM-X",
    "srtm_dem": "SRTM",
    "fab_dem": "FABDEM",
    "copernicus_dem": "Copernicus DEM",   # <- правильний ключ
    "nasa_dem": "NASADEM",
    "alos_dem": "ALOS",
    "aster_dem": "ASTER",
}
DEM_LIST: List[str] = ["alos_dem", "aster_dem", "copernicus_dem", "fab_dem", "nasa_dem", "srtm_dem", "tan_dem"]
COLORMAPS = ["viridis", "terrain" ]
MAP_STYLES = {
    "Outdoors": "mapbox://styles/mapbox/outdoors-v12",
    "Satellite Streets": "mapbox://styles/mapbox/satellite-streets-v12",
}

# ---------- Basin (vector overlay; GeoJSON expects EPSG:4326) ----------
try:
    basin: gpd.GeoDataFrame = get_df("basin")
    basin = basin.to_crs("EPSG:4326")
    BASIN_JSON = json.loads(basin.to_json())
    logger.info("Basin OK: CRS=%s, rows=%d", basin.crs, len(basin))
except Exception as e:
    logger.warning("Basin not available: %s", e)
    BASIN_JSON = None

# ---------- Helpers ----------
def _fix_path(p: str) -> str:
    if not p: return p
    p = p.replace("\\", "/")
    if p.startswith("/"): return os.path.normpath(p)
    if p.startswith("data/COG/"):  return "/app/data/cogs/" + p.split("data/COG/")[1]
    if p.startswith("data/cogs/"): return "/app/data/cogs/" + p.split("data/cogs/")[1]
    if p.startswith("data/"):      return "/app/" + p
    return os.path.normpath(p)

def _parse_level(s: str) -> int:
    try: return int(str(s).lower().replace("m", "").strip())
    except Exception: return 0

def build_dem_url(dem_name: str, cmap: str, stretch) -> str:
    s = f"[{stretch[0]},{stretch[1]}]"
    return f"{TC_BASE}/singleband/dem/{dem_name}" + "/{z}/{x}/{y}.png" + f"?colormap={cmap}&stretch_range={s}"

def build_flood_url(dem_name: str, hand_name: str, level: str, cmap: str, stretch, pure_blue: bool) -> str:
    # Terracotta tile name is in EPSG:3857 XYZ; deck.gl will place it correctly.
    layer = f"{dem_name}_{hand_name}_flood_{level}"
    s = f"[{stretch[0]},{stretch[1]}]"
    base = f"{TC_BASE}/singleband/flood_scenarios/{layer}" + "/{z}/{x}/{y}.png"
    return f"{base}?colormap=custom&colors=0000ff&stretch_range={s}" if pure_blue \
           else f"{base}?colormap={cmap}&stretch_range={s}"

# ---- deck.gl builders (no reprojection for tiles) ----
# --- 1) функція-саблейер у реєстрі компонента ---
BITMAP_FN = """
(props) => new deck.BitmapLayer({
  id: `${props.id}-bitmap`,
  image: props.tile.data,
  bounds: props.tile.bbox,
  opacity: props.opacity ?? 1,
  visible: props.visible ?? true,
  parameters: { depthTest: false }
})
"""

# --- 2) будівельник TileLayer: тільки посилання на ім'я функції ---
def tile_layer(layer_id: str, url: str, opacity: float = 1.0, visible: bool = True, z: int = 0, **_) -> dict:
    return {
        "@@type": "TileLayer",
        "id": layer_id,
        "data": url,
        "visible": visible,
        "minZoom": 0,
        "maxZoom": 19,
        "tileSize": 256,
        "opacity": opacity,
        "parameters": {"depthTest": False},
        "zIndex": z,  # порядок все одно визначає послідовність у "layers"
        "renderSubLayers": {"@@function": "bitmapTile"},
    }
def geojson_layer(data: dict, visible: bool = True, z: int = 0) -> dict:
    """Builds a deck.gl GeoJsonLayer dictionary."""
    return {
        "@@type": "GeoJsonLayer",
        "id": "basin-geojson",
        "data": data,
        "visible": visible,
        "filled": True,
        "stroked": True,
        "getFillColor": [30, 144, 255, 60],  # Dodger Blue, semi-transparent
        "getLineColor": [30, 144, 255, 200], # Dodger Blue, more opaque
        "getLineWidth": 2,
        "lineWidthUnits": "pixels",
        "zIndex": z,
    }
# --- 3) spec повертаємо як dict, БЕЗ жодних "functions" усередині ---
def build_spec(map_style, dem_url, flood_url, show_dem, show_flood, show_basin, basin_geojson):
    layers = []
    if dem_url:
        layers.append(tile_layer("dem-tiles", dem_url, opacity=0.75, visible=show_dem, z=10))
    if flood_url:
        layers.append(tile_layer("flood-tiles", flood_url, opacity=1.0, visible=show_flood, z=20))
    if basin_geojson: # Use the argument passed to the function
        # Call the new helper function to build the layer
        layers.append(geojson_layer(basin_geojson, visible=show_basin, z=30))

    return {
        "mapStyle": map_style if MAPBOX_ACCESS_TOKEN else None,
        "initialViewState": {"longitude": 25.03, "latitude": 47.8, "zoom": 10, "pitch": 0, "bearing": 0},
        "layers": layers,
    }

# --- 4) у layout передаємо реєстр функцій у сам компонент ---
# deck = dash_deckgl.DashDeckgl(
#     id="deck-flood",
#     spec=build_spec(MAP_STYLES["Satellite Streets"], None, None, True, True, True, BASIN_JSON),
#     custom_libraries=[
#         {
#             "libraryName": "BitmapTileLibrary",
#             "libraryUrl": "", # Not needed for inline functions
#             "functions": {
#                 "bitmapTile": BITMAP_FN
#             }
#         }
#     ],
#     height=700,
#     mapbox_key=MAPBOX_ACCESS_TOKEN,
#     cursor_position="bottom-right",
#     events=[],
# )

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
    logger.info("layers_index normalized: %d entries", len(layers_index))
except Exception as e:
    logger.warning("Failed to read %s: %s", ASSETS_INDEX_PATH, e)
    layers_index = []

# Build: DEM_LEVELS[dem] -> ["1m",...]; DEM_LEVEL_TO_HAND[dem][level] -> "hand_2000" (prefer) or first
DEM_LEVELS: Dict[str, List[str]] = {}
DEM_LEVEL_TO_HAND: Dict[str, Dict[str, str]] = {}
if layers_index:
    tmp_levels: Dict[str, set] = {}
    tmp_level2hand: Dict[Tuple[str, str], set] = {}
    for r in layers_index:
        if r.get("category") != "flood_scenarios":
            continue
        dem, hand, level = r.get("dem"), r.get("hand"), r.get("flood")
        if not (dem and hand and level): continue
        tmp_levels.setdefault(dem, set()).add(level)
        tmp_level2hand.setdefault((dem, level), set()).add(hand)
    for dem, levels_set in tmp_levels.items():
        levels_sorted = sorted(levels_set, key=_parse_level)
        DEM_LEVELS[dem] = levels_sorted
        DEM_LEVEL_TO_HAND[dem] = {}
        for lvl in levels_sorted:
            hands = list(tmp_level2hand.get((dem, lvl), []))
            DEM_LEVEL_TO_HAND[dem][lvl] = "hand_2000" if "hand_2000" in hands else (hands[0] if hands else "")
    DEM_LIST = sorted(DEM_LEVELS.keys())

logger.info("DEMs: %s", ", ".join(DEM_LIST))

# ---------- UI ----------

layout = html.Div([
    html.H3("Flood Scenarios (deck.gl)"),

    # Карта (обгорнута в Div зі стилем)
    html.Div([
        dash_deckgl.DashDeckgl(
            id="deck-flood",
            spec=build_spec(
                MAP_STYLES["Satellite Streets"],  # map_style
                None,  # dem_url
                None,  # flood_url
                True,  # show_dem
                True,  # show_flood
                True,  # show_basin
                BASIN_JSON  # basin_geojson
            ),
            custom_libraries=[{
                "libraryName": "BitmapTileLib",
                "libraryUrl": "",
                "functions": {"bitmapTile": BITMAP_FN}
            }],
            height=700,
            mapbox_key=MAPBOX_ACCESS_TOKEN,
            cursor_position="bottom-right",
            events=[],
        )
    ], style={"position": "relative", "zIndex": 10}),

    # Панель керування
    html.Div([
        html.Div([
            html.Label("DEM"),
            dcc.Dropdown(
                id="dem-name",
                options=[{"label": DEM_LABELS.get(d, d), "value": d} for d in DEM_LIST],
                value=DEM_LIST[0], style={"width": 220}, clearable=False
            ),
        ], style={"display": "inline-block", "marginRight": 12}),

        html.Div([
            html.Label("Flood level (m)"),
            dcc.Dropdown(id="flood-level", options=[], placeholder="1–10 m",
                         style={"width": 160}, clearable=False),
        ], style={"display": "inline-block", "marginRight": 12}),

        html.Div([
            html.Label("DEM Colormap"),
            dcc.Dropdown(id="dem-cmap",
                         options=[{"label": c.capitalize(), "value": c} for c in COLORMAPS],
                         value="terrain", style={"width": 150}, clearable=False),
        ], style={"display": "inline-block", "marginRight": 12}),

        html.Div([
            html.Label("DEM Stretch"),
            dcc.RangeSlider(id="dem-stretch", min=0, max=4000, step=50, value=[250, 2200],
                            marks={0:"0", 1000:"1000", 2000:"2000", 3000:"3000", 4000:"4000"},
                            tooltip={"always_visible": False, "placement": "top"}),
        ], style={"display": "inline-block", "width": 360, "marginRight": 12}),

        html.Div([
            html.Label("Flood Colormap"),
            dcc.Dropdown(id="flood-cmap",
                         options=[{"label": "Blues", "value": "blues"},
                                  {"label": "Viridis", "value": "viridis"},
                                  {"label": "Pure Blue", "value": "custom"}],
                         value="blues", style={"width": 150}, clearable=False),
        ], style={"display": "inline-block", "marginRight": 12}),

        html.Div([
            html.Label("Flood Stretch"),
            dcc.RangeSlider(id="flood-stretch", min=0, max=10, step=1, value=[0, 5],
                            marks={i: str(i) for i in range(11)},
                            tooltip={"always_visible": False, "placement": "top"}),
        ], style={"display": "inline-block", "width": 260}),

        html.Div([
            html.Label("Map style"),
            dcc.Dropdown(id="map-style",
                         options=[{"label": k, "value": v} for k, v in MAP_STYLES.items()],
                         value=MAP_STYLES["Satellite Streets"],
                         style={"width": 260}, clearable=False),
        ], style={"display": "inline-block", "marginLeft": 12}),

        html.Div([
            html.Label("Overlays"),
            dcc.Checklist(
                id="overlay-toggle",
                options=[{"label": "Show DEM", "value": "dem"},
                         {"label": "Show Flood", "value": "flood"},
                         {"label": "Show Basin", "value": "basin"}],
                value=["dem", "flood", "basin"],
                inline=True
            ),
        ], style={"display": "inline-block", "marginLeft": 18}),
    ], style={"marginBottom": 12}),
])


# ---------- Callbacks ----------
@callback(
    Output("flood-level", "options"),
    Output("flood-level", "value"),
    Input("dem-name", "value")
)
def _update_levels(dem_name: str):
    levels = DEM_LEVELS.get(dem_name) or []
    opts = [{"label": f"{_parse_level(l)} m", "value": l} for l in levels]
    default = "5m" if "5m" in levels else (levels[0] if levels else None)
    logger.info("[levels] dem=%s -> levels=%s; default=%s", dem_name, levels, default)
    return opts, default

@callback(
    Output("deck-flood", "spec"),
    Input("dem-name", "value"),
    Input("dem-cmap", "value"),
    Input("dem-stretch", "value"),
    Input("flood-level", "value"),
    Input("flood-cmap", "value"),
    Input("flood-stretch", "value"),
    Input("map-style", "value"),
    Input("overlay-toggle", "value"),
)
def _update_spec(dem_name, dem_cmap, dem_stretch,
                 flood_level, flood_cmap, flood_stretch,
                 map_style, overlays):

    # ---- defaults: True, якщо чекліст ще None
    if overlays is None:
        show_dem = show_flood = show_basin = True
    else:
        show_dem = "dem" in overlays
        show_flood = "flood" in overlays
        show_basin = "basin" in overlays

    dem_url = build_dem_url(dem_name, dem_cmap or "terrain", dem_stretch or [250, 2200]) if dem_name else ""
    flood_url, chosen_hand = "", ""
    if dem_name and flood_level:
        chosen_hand = (DEM_LEVEL_TO_HAND.get(dem_name, {}) or {}).get(flood_level, "")
        if chosen_hand:
            flood_url = build_flood_url(
                dem_name, chosen_hand, flood_level,
                flood_cmap or "blues",
                flood_stretch or [0, 5],
                pure_blue=(flood_cmap == "custom")
            )

    # ---- console logs
    logger.info("[spec.in] dem=%s cmap=%s stretch=%s level=%s flood_cmap=%s flood_stretch=%s",
                dem_name, dem_cmap, dem_stretch, flood_level, flood_cmap, flood_stretch)
    if dem_name:
        logger.info("[DEM] visible=%s url=%s", show_dem, dem_url)
    if flood_level:
        if flood_url:
            layer_name = f"{dem_name}_{chosen_hand}_flood_{flood_level}"
            logger.info("[FLOOD] layer=%s visible=%s url=%s", layer_name, show_flood, flood_url)
        else:
            reason = ("no matching hand" if not chosen_hand else "url not built")
            logger.info("[FLOOD] not shown (%s) dem=%s level=%s", reason, dem_name, flood_level)
    logger.info("[MAP] style=%s basin=%s", map_style, show_basin)

    spec = build_spec(map_style, dem_url or None, flood_url or None,
                      show_dem, show_flood, show_basin, BASIN_JSON)
    return spec
