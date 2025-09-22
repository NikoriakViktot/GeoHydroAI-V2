# pages/flood_map_deck.py
import os, json, logging
from typing import Dict, List, Tuple

import geopandas as gpd
import dash
from dash import html, dcc, callback, Output, Input, State, no_update
import dash_deckgl

from registry import get_df

# ---------- Page & logging ----------
dash.register_page(__name__, path="/flood-map", name="Flood Scenarios (deck.gl)", order=98)
app = dash.get_app()

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=os.getenv("PAGE_LOG_LEVEL", "INFO"),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

TC_BASE = os.getenv("TERRACOTTA_PUBLIC_URL", "https://www.geohydroai.org/tc").rstrip("/")
MAPBOX_ACCESS_TOKEN = os.getenv("MAPBOX_ACCESS_TOKEN", "").strip()
ASSETS_INDEX_PATH = "assets/layers_index.json"

# ---------- Basin (optional) ----------
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
    try:
        return int(str(s).lower().replace("m", "").strip())
    except Exception:
        return 0

def build_dem_url(dem_name: str, cmap: str, stretch) -> str:
    s = f"[{stretch[0]},{stretch[1]}]"
    return f"{TC_BASE}/singleband/dem/{dem_name}" + "/{z}/{x}/{y}.png" + f"?colormap={cmap}&stretch_range={s}"

def build_flood_url(dem_name: str, hand_name: str, level: str, cmap: str, stretch, pure_blue: bool) -> str:
    layer = f"{dem_name}_{hand_name}_flood_{level}"
    s = f"[{stretch[0]},{stretch[1]}]"
    base = f"{TC_BASE}/singleband/flood_scenarios/{layer}" + "/{z}/{x}/{y}.png"
    return f"{base}?colormap=custom&colors=0000ff&stretch_range={s}" if pure_blue \
           else f"{base}?colormap={cmap}&stretch_range={s}"

# ---- deck.gl builders
def tile_layer(layer_id: str, url: str, opacity: float = 1.0, visible: bool = True) -> dict:
    return {
        "@@type": "TileLayer",
        "id": layer_id,
        "data": url,
        "visible": visible,
        "minZoom": 0, "maxZoom": 19, "tileSize": 256, "opacity": opacity,
        "renderSubLayers": {
            "@@function": ["tile", {
                "type": "BitmapLayer",
                "id": f"{layer_id}-bitmap",
                "image": "@@tile.data",
                "bounds": "@@tile.bbox",
                "opacity": opacity,
                "visible": visible
            }]
        },
    }

def basin_layer(geojson: dict, visible: bool = True) -> dict:
    return {
        "@@type": "GeoJsonLayer",
        "id": "basin-outline",
        "data": geojson,
        "stroked": True, "filled": False,
        "getLineColor": [0, 102, 255, 200], "getLineWidth": 2, "lineWidthUnits": "pixels",
        "visible": visible
    }

def build_spec(map_style: str,
               dem_url: str | None, flood_url: str | None,
               show_dem: bool, show_flood: bool, show_basin: bool,
               basin_geojson: dict | None) -> str:
    layers = []
    if dem_url:
        layers.append(tile_layer("dem-tiles", dem_url, opacity=0.75, visible=show_dem))
    if flood_url:
        layers.append(tile_layer("flood-tiles", flood_url, opacity=1.0, visible=show_flood))
    if basin_geojson:
        layers.append(basin_layer(basin_geojson, visible=show_basin))

    spec = {
        "mapStyle": map_style if MAPBOX_ACCESS_TOKEN else None,
        "initialViewState": {"longitude": 25.03, "latitude": 47.8, "zoom": 10, "pitch": 0, "bearing": 0},
        "layers": layers
    }
    return json.dumps(spec)

# ---------- Read & normalize layers_index.json ----------
# DEM labels (UI)
DEM_LABELS = {
    "tan_dem": "TanDEM-X",
    "srtm_dem": "SRTM",
    "fab_dem": "FABDEM",
    "copernicus_dem": "Copernicus DEM",  # виправлена назва ключа
    "nasa_dem": "NASADEM",
    "alos_dem": "ALOS",
    "aster_dem": "ASTER",
}

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
    logger.info("layers_index normalized: %d entries", len(layers_index))
except Exception as e:
    logger.warning("Failed to read %s: %s", ASSETS_INDEX_PATH, e)
    layers_index = []

# Build:
#  - DEM_LEVELS[dem] -> ["1m","2m",...]
#  - DEM_LEVEL_TO_HAND[dem][level] -> "hand_2000" (або перший доступний)
DEM_LEVELS: Dict[str, List[str]] = {}
DEM_LEVEL_TO_HAND: Dict[str, Dict[str, str]] = {}

if layers_index:
    tmp_levels: Dict[str, set] = {}
    tmp_level2hand: Dict[Tuple[str, str], set] = {}

    for r in layers_index:
        if r.get("category") != "flood_scenarios":
            continue
        dem = r.get("dem"); hand = r.get("hand"); level = r.get("flood")
        if not (dem and hand and level): continue
        tmp_levels.setdefault(dem, set()).add(level)
        tmp_level2hand.setdefault((dem, level), set()).add(hand)

    for dem, levels_set in tmp_levels.items():
        levels_sorted = sorted(levels_set, key=_parse_level)
        DEM_LEVELS[dem] = levels_sorted
        DEM_LEVEL_TO_HAND[dem] = {}
        for lvl in levels_sorted:
            hands = list(tmp_level2hand.get((dem, lvl), []))
            # пріоритет: hand_2000 → інакше перший
            chosen = "hand_2000" if "hand_2000" in hands else (hands[0] if hands else "")
            DEM_LEVEL_TO_HAND[dem][lvl] = chosen

    DEM_LIST = sorted(DEM_LEVELS.keys())

logger.info("DEMs: %s", ", ".join(DEM_LIST))

# ---------- UI ----------
COLORMAPS = ["viridis", "terrain", "inferno", "magma", "plasma"]
MAP_STYLES = {
    "Outdoors": "mapbox://styles/mapbox/outdoors-v12",
    "Satellite Streets": "mapbox://styles/mapbox/satellite-streets-v12",
}

layout = html.Div([
    html.H3("Flood Scenarios (deck.gl)"),

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

    dash_deckgl.DashDeckgl(
        id="deck-flood",
        spec=build_spec(MAP_STYLES["Satellite Streets"], None, None, True, True, True, BASIN_JSON),
        height=700,
        mapbox_key=MAPBOX_ACCESS_TOKEN,
        cursor_position="bottom-right",
        events=[],
    ),

    html.Div(id="deck-log", style={"fontFamily": "monospace", "marginTop": "6px"}),
])

# ---------- Callbacks ----------
@callback(
    Output("flood-level", "options"),
    Output("flood-level", "value"),
    Input("dem-name", "value"),
)
def _update_levels(dem_name: str):
    levels = DEM_LEVELS.get(dem_name) or []
    opts = [{"label": f"{_parse_level(l)} m", "value": l} for l in levels]
    default = "5m" if "5m" in levels else (levels[0] if levels else None)
    logger.info("[levels] dem=%s -> levels=%s; default=%s", dem_name, levels, default)
    return opts, default

@callback(
    Output("deck-flood", "spec"),
    Output("deck-log", "children"),
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
    # visible flags
    show_dem = "dem" in (overlays or [])
    show_flood = "flood" in (overlays or [])
    show_basin = "basin" in (overlays or [])

    # DEM URL
    dem_url = build_dem_url(dem_name, dem_cmap or "viridis", dem_stretch or [250, 2200]) if dem_name else ""

    # FLOOD URL
    flood_url = ""
    chosen_hand = ""
    if dem_name and flood_level:
        chosen_hand = (DEM_LEVEL_TO_HAND.get(dem_name, {}) or {}).get(flood_level, "")
        if chosen_hand:
            flood_url = build_flood_url(
                dem_name, chosen_hand, flood_level,
                flood_cmap or "blues",
                flood_stretch or [0, 5],
                pure_blue=(flood_cmap == "custom")
            )

    # ---- Logging to container
    if not dem_name:
        logger.info("[spec] NO DEM selected")
    else:
        logger.info("[spec] DEM=%s url=%s visible=%s", dem_name, dem_url, show_dem)

    if flood_level and chosen_hand and flood_url:
        layer_name = f"{dem_name}_{chosen_hand}_flood_{flood_level}"
        logger.info("[spec] FLOOD layer=%s url=%s visible=%s", layer_name, flood_url, show_flood)
    else:
        reason = ("no dem" if not dem_name else
                  "no level" if not flood_level else
                  "no matching hand" if not chosen_hand else
                  "no url")
        logger.info("[spec] NO FLOOD (%s) for dem=%s level=%s", reason, dem_name, flood_level)

    logger.info("[spec] map_style=%s show_dem=%s show_flood=%s show_basin=%s", map_style, show_dem, show_flood, show_basin)

    # ---- Log to UI
    ui_log = [
        f"DEM: {dem_name} ({DEM_LABELS.get(dem_name, dem_name)}) visible={show_dem}",
        f"DEM URL: {dem_url}",
        f"Flood: level={flood_level}, hand={chosen_hand or '—'}, visible={show_flood}",
        f"Flood URL: {flood_url or '—'}",
        f"Map style: {map_style}",
        f"Basin visible: {show_basin}"
    ]
    return build_spec(map_style, dem_url or None, flood_url or None, show_dem, show_flood, show_basin, BASIN_JSON), html.Pre("\n".join(ui_log))
