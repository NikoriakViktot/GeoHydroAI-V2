import os, json, uuid, time, logging
from collections import defaultdict
import numpy as np
import geopandas as gpd
import dash
from dash import html, dcc, callback, Output, Input, State, no_update
import dash_deckgl

from debug_utils import trace
from registry import get_df
from utils.dem_tools import (
    compute_dem_difference,
    make_colorbar_datauri,       # (не використовується у мін. версії, можна лишити)
    plot_histogram,              # (не використовується у мін. версії, можна лишити)
    calculate_error_statistics,  # (не використовується у мін. версії, можна лишити)
    diff_to_base64_png,
    raster_bounds_ll,
)
app = dash.get_app()

MAIN_MAP_HEIGHT = 550
RIGHT_PANEL_WIDTH = 400

logging.basicConfig(level=os.getenv("PAGE_LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

TC_BASE = os.getenv("TERRACOTTA_PUBLIC_URL", "https://www.geohydroai.org/tc").rstrip("/")
MAPBOX_ACCESS_TOKEN = os.getenv("MAPBOX_ACCESS_TOKEN", "").strip() or None

def safe_map_style() -> str | None:
    # Якщо токен є — використовуємо mapbox стиль; інакше None (без бекграунду)
    return "mapbox://styles/mapbox/light-v11" if MAPBOX_ACCESS_TOKEN else None

dash.register_page(__name__, path="/dem-diff", name="DEM Diff (deck.gl)", order=2)

# Basin
try:
    basin: gpd.GeoDataFrame = get_df("basin")
    basin = basin.to_crs("EPSG:4326")
    BASIN_JSON = json.loads(basin.to_json())
    logger.info("Basin loaded, CRS=%s, rows=%d", basin.crs, len(basin))
except Exception as e:
    logger.exception("Failed to load basin: %s", e)
    BASIN_JSON = None

# layers_index.json
ASSETS_INDEX_PATH = "assets/layers_index.json"
with open(ASSETS_INDEX_PATH, "r") as f:
    raw_index = json.load(f)

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

layers_index = []
for rec in raw_index:
    r = dict(rec)
    if r.get("path"):
        r["path"] = _fix_path(r["path"])
    layers_index.append(r)

# Групування
by_dem = defaultdict(list); categories = set()
for l in layers_index:
    by_dem[l.get("dem")].append(l)
    if l.get("category"): categories.add(l["category"])

DEM_LIST = sorted([d for d in by_dem.keys() if d])
CATEGORY_LIST = sorted(categories)

# ---- deck.gl helpers
def tile_layer(layer_id: str, url: str, opacity: float = 1.0) -> dict:
    return {
        "@@type": "TileLayer",
        "id": layer_id,
        "data": url,
        "minZoom": 0, "maxZoom": 19, "tileSize": 220, "opacity": opacity,
        "renderSubLayers": {
            "@@function": ["tile", {
                "type":"BitmapLayer",
                "id": f"{layer_id}-bitmap",
                "image":"@@tile.data", "bounds":"@@tile.bbox", "opacity": opacity
            }]
        },
    }

def bitmap_layer(layer_id: str, image_data_uri: str, bounds) -> dict:
    # bounds з raster_bounds_ll: [[S,W],[N,E]] -> [W,S,E,N]
    (south, west), (north, east) = bounds
    return {
        "@@type": "BitmapLayer",
        "id": layer_id,
        "image": image_data_uri,
        "bounds": [west, south, east, north],
        "opacity": 0.95,
        "parameters": {"depthTest": False},
    }

def basin_layer(geojson: dict) -> dict:
    return {
        "@@type":"GeoJsonLayer","id":"basin-outline",
        "data": geojson, "stroked": True, "filled": False,
        "getLineColor":[0,102,255,200], "getLineWidth":2, "lineWidthUnits":"pixels"
    }

def build_dem_url(colormap="viridis"):
    return f"{TC_BASE}/singleband/dem/fab_dem/{{z}}/{{x}}/{{y}}.png?colormap={colormap}&stretch_range=[250,2200]"

def build_spec(
    dem_url: str | None,
    diff_bitmap: dict | None,
    basin: dict | None,
    init_view=None,
    map_style: str | None = None
) -> dict:
    layers: list[dict] = []
    if dem_url:     layers.append(tile_layer("dem-tiles", dem_url, opacity=0.75))
    if diff_bitmap: layers.append(diff_bitmap)
    if basin:       layers.append(basin_layer(basin))
    return {
        "mapStyle": map_style,
        "initialViewState": init_view or {"longitude":25.03,"latitude":47.8,"zoom":8,"pitch":0,"bearing":0},
        # HUD тимчасово вимкнено, щоб відновити базовий рендер
        "layers": layers
    }

# ---- UI
layout = html.Div([
    html.H3("DEM Difference Analysis (deck.gl + Terracotta)"),
    dcc.Store(id="state-store", data={"vmin": None, "vmax": None, "cmap": "RdBu_r"}),
    dcc.Store(id="debug-log", data=[]),
    html.Div([
        html.Div([
            html.Label("DEM 1"),
            dcc.Dropdown(id="dem1", options=[{"label": d, "value": d} for d in DEM_LIST]),
            html.Label("DEM 2"),
            dcc.Dropdown(id="dem2", options=[{"label": d, "value": d} for d in DEM_LIST]),
            html.Label("Категорія"),
            dcc.Dropdown(id="cat", options=[{"label": c, "value": c} for c in CATEGORY_LIST]),
            html.Br(),
            html.Button("Порахувати різницю", id="run"),
        ], style={"width": "300px","minWidth": "280px","paddingRight": "16px"}),

        html.Div([
            dash_deckgl.DashDeckgl(
                id="deck-main",
                spec=build_spec(build_dem_url("viridis"), None, BASIN_JSON, map_style=safe_map_style()),
                height=MAIN_MAP_HEIGHT,
                cursor_position="bottom-right",
                events=["hover", "click"],   # події лишили, але без мутацій spec
                mapbox_key=MAPBOX_ACCESS_TOKEN,
            ),
            html.Div(id="deck-events", style={"fontFamily": "monospace", "marginTop": "6px"})
        ], style={"width": "100%"}),
    ], style={"display": "grid","gridTemplateColumns":"300px 1fr","gap":"8px","alignItems":"start"}),
])

# ---- helpers
def _pick_path(name, category):
    arr = by_dem.get(name, [])
    if not arr: return None
    if category:
        for it in arr:
            if it.get("category") == category:
                return it.get("path")
    return arr[0].get("path")

def _rid(): return uuid.uuid4().hex[:8]

# ---- main callback: мінімальний рендер без HUD
@app.callback(
    Output("deck-main","spec"),
    Output("state-store","data"),
    Output("debug-log","data"),
    Input("run","n_clicks"),
    State("dem1","value"), State("dem2","value"), State("cat","value"),
    State("debug-log","data"),
    prevent_initial_call=True
)
@trace("run_diff")
def run_diff(n, dem1, dem2, cat, dbg):
    rid = _rid()
    t0 = time.perf_counter()
    dbg = list(dbg or [])
    def push(msg):
        line = f"[{rid}] {msg}"; logger.info(line); dbg.append(line)

    push(f"n={n} dem1={dem1} dem2={dem2} cat={cat}")

    if not dem1 or not dem2 or dem1 == dem2:
        push("invalid selection")
        return no_update, no_update, dbg

    p1, p2 = _pick_path(dem1, cat), _pick_path(dem2, cat)
    p1, p2 = _fix_path(p1), _fix_path(p2)
    push(f"paths {p1} | {p2}")

    try:
        diff, ref = compute_dem_difference(p1, p2)
        nan_pct = float(np.isnan(diff).mean() * 100.0)
        push(f"diff shape={getattr(diff,'shape',None)} NaN={nan_pct:.2f}%")
    except Exception as e:
        push(f"compute FAIL: {e}")
        return no_update, no_update, dbg

    try:
        if (cat or "").lower() == "dem":
            vmin, vmax = -25.0, 25.0
            push("stretch fixed [-25,25]")
        else:
            q1, q99 = np.nanpercentile(diff, [1, 99])
            vmin, vmax = float(q1), float(q99)
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin, vmax = -10.0, 10.0
            push(f"stretch pct [{vmin:.3f},{vmax:.3f}]")
    except Exception:
        vmin, vmax = -10.0, 10.0
        push("stretch fallback [-10,10]")

    try:
        img_uri = diff_to_base64_png(diff, ref, vmin=vmin, vmax=vmax, figsize=(8, 8))
        bounds  = raster_bounds_ll(ref)
        diff_bitmap = bitmap_layer("diff-bitmap", img_uri, bounds)
        push("overlay ok")
    except Exception as e:
        push(f"overlay FAIL: {e}")
        return no_update, no_update, dbg

    # мінімальний spec без HUD
    spec_obj = build_spec(build_dem_url("terrain"), diff_bitmap, BASIN_JSON, map_style=safe_map_style())

    state = {"vmin": vmin, "vmax": vmax, "cmap": "RdBu_r", "dem1": dem1, "dem2": dem2, "cat": cat}
    push(f"done {(time.perf_counter()-t0)*1000:.1f} ms")
    return spec_obj, state, dbg
