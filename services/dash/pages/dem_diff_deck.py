# pages/dem_diff_deck.py
import os, json, logging
from collections import defaultdict
import numpy as np
import geopandas as gpd
import dash
from dash import html, dcc, callback, Output, Input, State, no_update
import dash_deckgl

from utils.dem_tools import (
    compute_dem_difference, make_colorbar_datauri,
    plot_histogram, calculate_error_statistics,
    diff_to_base64_png, raster_bounds_ll,   # <-- додано
)
from registry import get_df
MAIN_MAP_HEIGHT = 500
RIGHT_PANEL_WIDTH = 500
COMPARE_MAP_HEIGHT = 360
# ---- Логи (детальні)
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=os.getenv("PAGE_LOG_LEVEL", "INFO"),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger.info("=== dem_diff_deck page init ===")

TC_BASE = os.getenv("TERRACOTTA_PUBLIC_URL", "https://www.geohydroai.org/tc").rstrip("/")
MAPBOX_ACCESS_TOKEN = os.getenv("MAPBOX_ACCESS_TOKEN", "").strip()

dash.register_page(__name__, path="/dem-diff", name="DEM Diff (deck.gl)", order=2)

# ---- Basin
try:
    basin: gpd.GeoDataFrame = get_df("basin")
    logger.info("Basin loaded, CRS=%s, rows=%d", basin.crs, len(basin))
    basin = basin.to_crs("EPSG:4326")
    basin_json = json.loads(basin.to_json())
except Exception as e:
    logger.exception("Failed to load basin: %s", e)
    basin_json = None

# ---- layers_index.json (як у тебе; якщо є _fix_path — лишаємо)
ASSETS_INDEX_PATH = "assets/layers_index.json"
with open(ASSETS_INDEX_PATH, "r") as f:
    raw_index = json.load(f)

def _fix_path(p: str) -> str:
    if not p: return p
    p = p.replace("\\", "/")
    if p.startswith("/"): return os.path.normpath(p)
    if p.startswith("data/COG/"):  return "/app/data/cogs/" + p.split("data/COG/")[1]
    if p.startswith("data/cogs/"): return "/app/data/cogs/" + p.split("data/cogs/")[1]
    if p.startswith("data/"):      return "/app/" + p
    return os.path.normpath(p)

layers_index = []
for rec in raw_index:
    r = dict(rec)
    if r.get("path"):
        r["path"] = _fix_path(r["path"])
    layers_index.append(r)
logger.info("Layers index loaded: %d entries", len(layers_index))

# ---- Групування
by_dem = defaultdict(list); categories = set()
for l in layers_index:
    by_dem[l.get("dem")].append(l)
    if l.get("category"): categories.add(l["category"])
DEM_LIST = sorted([d for d in by_dem.keys() if d])
CATEGORY_LIST = sorted(categories)

# ===== UI налаштування висот


# ---- deck.gl helper-и
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
        "opacity": 0.95
    }

def basin_layer(geojson: dict) -> dict:
    return {
        "@@type":"GeoJsonLayer","id":"basin-outline",
        "data": geojson, "stroked": True, "filled": False,
        "getLineColor":[0,102,255,200], "getLineWidth":2, "lineWidthUnits":"pixels"
    }

def build_dem_url(colormap="viridis"):
    return f"{TC_BASE}/singleband/dem/fab_dem/{{z}}/{{x}}/{{y}}.png?colormap={colormap}&stretch_range=[250,2200]"

def build_spec(dem_url: str | None, diff_bitmap: dict | None, basin: dict | None,
               init_view=None, map_style="mapbox://styles/mapbox/light-v11") -> str:
    layers = []
    if dem_url:     layers.append(tile_layer("dem-tiles", dem_url, opacity=0.75))
    if diff_bitmap: layers.append(diff_bitmap)  # <-- головне місце
    if basin:       layers.append(basin_layer(basin))
    spec = {
        "mapStyle": map_style,
        "initialViewState": init_view or {"longitude":25.03,"latitude":47.8,"zoom":9,"pitch":0,"bearing":0},
        "layers": layers
    }
    return json.dumps(spec)
layout = html.Div([
    html.H3("DEM Difference Analysis (deck.gl + Terracotta)"),

    # Головна сітка: ліворуч вузька колонка з контролями, праворуч — карта+права панель
    html.Div([
        # ---- Ліва панель керування ----
        html.Div([
            html.Label("DEM 1"),
            dcc.Dropdown(id="dem1", options=[{"label": d, "value": d} for d in DEM_LIST]),
            html.Label("DEM 2"),
            dcc.Dropdown(id="dem2", options=[{"label": d, "value": d} for d in DEM_LIST]),
            html.Label("Категорія"),
            dcc.Dropdown(id="cat", options=[{"label": c, "value": c} for c in CATEGORY_LIST]),
            html.Br(),
            html.Button("Порахувати різницю", id="run"),
        ], style={
            "width": "300px",
            "minWidth": "280px",
            "paddingRight": "16px"
        }),

        # ---- Права зона: карта + панель справа ----
        html.Div([
            # внутрішня сітка: карта (розтягується) + права панель фіксованої ширини
            html.Div([
                # Карта
                dash_deckgl.DashDeckgl(
                    id="deck-main",
                    spec=build_spec(build_dem_url("viridis"), None, basin_json),
                    description={"top-right": "<div id='legend'>Legend</div>"},
                    height=MAIN_MAP_HEIGHT,
                    cursor_position="bottom-right",
                    events=["hover"],
                    mapbox_key=MAPBOX_ACCESS_TOKEN,
                ),

                # Права панель: гістограма + статистика
                html.Div([
                    html.H4("Гістограма", style={"marginTop": 0}),
                    html.Img(id="hist", style={"width": "100%", "borderRadius": "8px"}),
                    html.Hr(),
                    html.Div(id="stats", style={"fontFamily": "monospace"})
                ], style={
                    "width": f"{RIGHT_PANEL_WIDTH}px",
                    "maxWidth": f"{RIGHT_PANEL_WIDTH}px",
                    "paddingLeft": "12px",
                    "overflowY": "auto"
                }),
            ], style={
                "display": "grid",
                "gridTemplateColumns": f"1fr {RIGHT_PANEL_WIDTH}px",
                "gap": "12px",
                "alignItems": "start"
            }),

            # Події під картою (за бажанням)
            html.Div(id="deck-events", style={"fontFamily": "monospace", "marginTop": "6px"})
        ], style={"width": "100%"})

    ], style={
        "display": "grid",
        "gridTemplateColumns": "300px 1fr",
        "gap": "8px",
        "alignItems": "start"
    })
])

# ---- Службові
def _pick_path(name, category):
    arr = by_dem.get(name, [])
    if not arr: return None
    if category:
        for it in arr:
            if it.get("category") == category:
                return it.get("path")
    return arr[0].get("path")

# ---- Колбек
@callback(
    Output("deck-main","spec"),
    Output("hist","src"),
    Output("stats","children"),
    Input("run","n_clicks"),
    State("dem1","value"), State("dem2","value"), State("cat","value"),
    prevent_initial_call=True
)
def run_diff(n, dem1, dem2, cat):
    logger.info("Run clicked: n=%s, dem1=%s, dem2=%s, cat=%s", n, dem1, dem2, cat)
    if not dem1 or not dem2 or dem1 == dem2:
        logger.warning("Invalid DEM selection")
        return no_update, no_update, "Оберіть різні DEM!"

    # шляхи
    p1, p2 = _pick_path(dem1, cat), _pick_path(dem2, cat)
    p1, p2 = _fix_path(p1), _fix_path(p2)
    logger.info("Using paths: %s | %s", p1, p2)

    # обчислення різниці
    try:
        diff, ref = compute_dem_difference(p1, p2)
        nan_pct = float(np.isnan(diff).mean() * 100.0)
        logger.info("Diff shape=%s, NaNs=%.2f%%", diff.shape, nan_pct)
    except Exception as e:
        logger.exception("Error computing diff: %s", e)
        return no_update, no_update, f"Помилка при обчисленні: {e}"

    # динамічний/фіксований діапазон відображення
    try:
        if (cat or "").lower() == "dem":
            vmin, vmax = -25.0, 25.0
            logger.info("Using fixed DEM stretch: [%.3f, %.3f]", vmin, vmax)
        else:
            q1, q99 = np.nanpercentile(diff, [1, 99])
            vmin, vmax = float(q1), float(q99)
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin, vmax = -10.0, 10.0
            logger.info("Percentile stretch: [%.3f, %.3f]", vmin, vmax)
    except Exception:
        vmin, vmax = -10.0, 10.0
        logger.info("Fallback stretch: [%.3f, %.3f]", vmin, vmax)

    # рендер diff -> data-URI + коректні bounds у WGS84
    try:
        img_uri = diff_to_base64_png(diff, ref, vmin=vmin, vmax=vmax, figsize=(8, 8))
        bounds = raster_bounds_ll(ref)  # [[S,W],[N,E]] у EPSG:4326
        logger.info("Overlay data-uri length=%d chars, bounds=%s", len(img_uri), bounds)
        diff_bitmap = bitmap_layer("diff-bitmap", img_uri, bounds)
    except Exception as e:
        logger.exception("Error building Bitmap overlay: %s", e)
        return no_update, no_update, f"Помилка рендерінгу накладання: {e}"

    # легенда / гістограма / статистика
    legend_uri = make_colorbar_datauri(vmin, vmax, cmap="RdBu_r")
    legend_html = f"<img src='{legend_uri}' style='height:160px'/>"  # ✅ виправлено

    hist_png = plot_histogram(diff, clip_range=(vmin, vmax))
    stats = calculate_error_statistics(diff)
    rows = [
        html.Tr([html.Th(k), html.Td(f"{v:.3f}" if isinstance(v, float) and np.isfinite(v) else v)])
        for k, v in stats.items()
    ]
    stats_tbl = html.Table(rows, style={"background": "#181818", "color": "#eee", "padding": "6px"})

    # збірка deck.gl spec
    spec = build_spec(build_dem_url("terrain"), diff_bitmap, basin_json)
    spec_obj = json.loads(spec)
    spec_obj.setdefault("description", {})["top-right"] = legend_html

    return json.dumps(spec_obj), hist_png, stats_tbl


@callback(Output("deck-events","children"), Input("deck-main","lastEvent"))
def show_evt(evt):
    if not evt: return ""
    return f"{evt.get('eventType')} @ {evt.get('coordinate')}"
