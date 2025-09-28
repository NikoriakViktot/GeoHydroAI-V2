# pages/dem_diff_deck.py
import os, json, logging
from collections import defaultdict
import numpy as np
import geopandas as gpd
from xdem import DEM as _DEM
import dash
from dash import html, dcc, callback, Output, Input, State, no_update
import dash_deckgl

from utils.dem_tools import (
    compute_dem_difference, make_colorbar_datauri, calculate_error_statistics,
    diff_to_base64_png, raster_bounds_ll,
    plotly_histogram_figure, read_binary_raster, pixel_area_m2_from_ref_dem,
    flood_metrics, plotly_flood_areas_figure, flood_compare_overlay_png
)

from registry import get_df

MAIN_MAP_HEIGHT = 550
RIGHT_PANEL_WIDTH = 500

# ---- Логи
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=os.getenv("PAGE_LOG_LEVEL", "INFO"),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger.info("=== dem_diff_deck page init ===")

TC_BASE = os.getenv("TERRACOTTA_PUBLIC_URL", "https://www.geohydroai.org/tc").rstrip("/")
MAPBOX_ACCESS_TOKEN = os.getenv("MAPBOX_ACCESS_TOKEN", "").strip()

dash.register_page(__name__, path="/dem-diff", name="DEM Diff", order=2)
app = dash.get_app()

# ---- Basin
try:
    basin: gpd.GeoDataFrame = get_df("basin")
    logger.info("Basin loaded, CRS=%s, rows=%d", basin.crs, len(basin))
    basin = basin.to_crs("EPSG:4326")
    basin_json = json.loads(basin.to_json())
except Exception as e:
    logger.exception("Failed to load basin: %s", e)
    basin_json = None

# ---- layers_index.json
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
logger.info("Layers index loaded: %d entries", len(layers_index))

# ---- Групування за DEM та категоріями
by_dem = defaultdict(list)
categories = set()
for l in layers_index:
    by_dem[l.get("dem")].append(l)
    if l.get("category"):
        categories.add(l["category"])
DEM_LIST = sorted([d for d in by_dem.keys() if d])
CATEGORY_LIST = sorted(categories)

# ---- Flood index (службове)
def _flood_level_key(s):
    if s is None:
        return float("inf")
    t = str(s).strip().lower().replace(" ", "")
    if t.endswith("m"):
        t = t[:-1]
    try:
        return float(t)
    except Exception:
        return float("inf")

def build_flood_index(layers_index):
    idx = {}
    flood_values = set()
    hand_values = set()
    dem_values = set()
    for rec in layers_index:
        if rec.get("category") != "flood_scenarios":
            continue
        key = (rec.get("dem"), rec.get("hand"), rec.get("flood"))
        if rec.get("path"):
            idx[key] = rec["path"]
        if rec.get("flood"):
            flood_values.add(rec["flood"])
        if rec.get("hand"):
            hand_values.add(rec["hand"])
        if rec.get("dem"):
            dem_values.add(rec["dem"])
    return idx, sorted(dem_values), sorted(hand_values), sorted(flood_values, key=_flood_level_key)

FLOOD_INDEX, FLOOD_DEMS, FLOOD_HANDS, FLOOD_LEVELS = build_flood_index(layers_index)

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
    if diff_bitmap: layers.append(diff_bitmap)
    if basin:       layers.append(basin_layer(basin))
    spec = {
        "mapStyle": map_style,
        "initialViewState": init_view or {"longitude":25.03,"latitude":47.8,"zoom":8,"pitch":0,"bearing":0},
        "layers": layers
    }
    return json.dumps(spec)

# ---- Layout
layout = html.Div([
    html.H3("DEM Difference Analysis"),

    html.Div([
        # Ліва панель
        html.Div([
            html.Label("DEM 1", style={"marginBottom": "4px", "fontWeight": "bold"}),
            dcc.Dropdown(id="dem1", options=[{"label": d, "value": d} for d in DEM_LIST],
                         style={"marginBottom": "10px", "fontSize": "14px"}),

            html.Label("DEM 2", style={"marginBottom": "4px", "fontWeight": "bold"}),
            dcc.Dropdown(id="dem2", options=[{"label": d, "value": d} for d in DEM_LIST],
                         style={"marginBottom": "10px", "fontSize": "14px"}),

            html.Label("Category", style={"marginBottom": "4px", "fontWeight": "bold"}),
            dcc.Dropdown(id="cat", options=[{"label": c, "value": c} for c in CATEGORY_LIST],
                         style={"marginBottom": "8px", "fontSize": "14px"}),

            # Параметри тільки для flood_scenarios:
            html.Div([
                html.Label("HAND model", style={"marginBottom": "4px"}),
                dcc.Dropdown(id="flood_hand", options=[{"label": h, "value": h} for h in FLOOD_HANDS],
                             value=(FLOOD_HANDS[0] if FLOOD_HANDS else None), style={"marginBottom": "8px"}),

                html.Label("Flood level", style={"marginBottom": "4px"}),
                dcc.Dropdown(id="flood_level", options=[{"label": f, "value": f} for f in FLOOD_LEVELS],
                             value=(FLOOD_LEVELS[0] if FLOOD_LEVELS else None)),
            ], id="flood_opts", style={"display": "none", "marginBottom": "10px"}),

            html.Button("Compute Difference", id="run", style={
                "backgroundColor": "#1f77b4", "color": "white", "border": "none", "borderRadius": "6px",
                "padding": "8px 16px", "cursor": "pointer", "fontWeight": "bold", "fontSize": "14px", "width": "100%"
            })
        ], style={"width": "220px", "padding": "12px", "backgroundColor": "#1e1e1e", "borderRadius": "8px"}),

        # Права зона: карта + права панель
        html.Div([
            # Карта
            html.Div([
                dash_deckgl.DashDeckgl(
                    id="deck-main",
                    spec=build_spec(build_dem_url("viridis"), None, basin_json),
                    description={"top-right": "<div id='legend'>Legend</div>"},
                    height=MAIN_MAP_HEIGHT,
                    cursor_position="bottom-right",
                    events=["hover"],
                    mapbox_key=MAPBOX_ACCESS_TOKEN,
                )
            ], style={
                "border": "1px solid rgba(255,255,255,0.15)",
                "borderRadius": "12px",
                "overflow": "hidden",
                "boxShadow": "0 4px 16px rgba(0,0,0,0.3)",
                "backgroundColor": "#111",
                "padding": "2px"
            }),

            # Права панель
            html.Div([
                html.H4("Histogram", style={"marginTop": 0}),
                dcc.Graph(
                    id="hist",
                    figure={},
                    style={"height": "260px"},
                    config={"displaylogo": False}
                ),
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

    ], style={
        "display": "grid",
        "gridTemplateColumns": "300px 1fr",
        "gap": "8px",
        "alignItems": "start"
    }),

    # Події під картою
    html.Div(id="deck-events", style={"fontFamily": "monospace", "marginTop": "6px"})
])

# ---- Службові
def _pick_path(name, category):
    arr = by_dem.get(name, [])
    if not arr:
        return None
    if category:
        for it in arr:
            if it.get("category") == category:
                return it.get("path")
    return arr[0].get("path")

@callback(Output("flood_opts","style"), Input("cat","value"))
def _toggle_flood_opts(cat):
    return {"display":"block","marginBottom":"10px"} if cat == "flood_scenarios" else {"display":"none"}

# ---- Колбек
@app.callback(
    Output("deck-main", "spec"),
    Output("hist", "figure"),
    Output("stats", "children"),
    Input("run", "n_clicks"),
    State("dem1", "value"), State("dem2", "value"), State("cat", "value"),
    State("flood_hand","value"), State("flood_level","value"),
    prevent_initial_call=True
)
def run_diff(n, dem1, dem2, cat, flood_hand, flood_level):
    logger.info("Run: dem1=%s dem2=%s cat=%s hand=%s level=%s", dem1, dem2, cat, flood_hand, flood_level)
    if not dem1 or not dem2 or dem1 == dem2:
        return no_update, no_update, "Please select two different DEMs."

    # Пошук шляху з метаданих (з пріоритезацією повних збігів для flood)
    def _find_path_for(dem_name, category, hand=None, level=None):
        cand = [r for r in by_dem.get(dem_name, []) if r.get("category")==category]
        if category == "flood_scenarios":
            if hand:
                cand = [r for r in cand if r.get("hand")==hand]
            if level:
                cand = [r for r in cand if r.get("flood")==level]
            if not cand:
                return None
            cand.sort(key=lambda r: (r.get("hand") is not None, r.get("flood") is not None), reverse=True)
            return _fix_path(cand[0]["path"])
        return _fix_path(cand[0]["path"]) if cand else None

    # --- FLOOD-порівняння ---
    if cat == "flood_scenarios":
        p1 = _find_path_for(dem1, "flood_scenarios", flood_hand, flood_level)
        p2 = _find_path_for(dem2, "flood_scenarios", flood_hand, flood_level)
        if not p1 or not p2:
            return no_update, no_update, "Flood layer not found for selected DEM/HAND/level."

        try:
            # бінарні маски затоплення
            A = read_binary_raster(p1)
            B = read_binary_raster(p2)

            # референтний DEM для bounds/площі пікселя
            base_dem_path = _pick_path(dem1, "dem") or _pick_path(dem1, None)
            if not base_dem_path:
                base_dem_path = _pick_path(dem2, "dem") or _pick_path(dem2, None) or p1
            base_dem_path = _fix_path(base_dem_path)

            ref = _DEM(base_dem_path)

            px_area = pixel_area_m2_from_ref_dem(ref)
            st = flood_metrics(A, B, px_area)

            # графік площ (Plotly)
            fig = plotly_flood_areas_figure(st, title=f"Flood {flood_hand} @ {flood_level}: areas & Δ")

            # overlay різниць
            overlay_uri = flood_compare_overlay_png(A, B, ref)
            bounds = raster_bounds_ll(ref)
            diff_bitmap = bitmap_layer("flood-diff-bitmap", overlay_uri, bounds)

            # легенда overlay
            legend_html = """
            <div style='background:#111;padding:6px 8px;border-radius:6px;color:#eee;font:12px/1.3 monospace'>
              <b>Overlay:</b>
              <span style='display:inline-block;width:10px;height:10px;background:#f55;opacity:0.8;margin:0 6px 0 10px;vertical-align:middle'></span>A only
              <span style='display:inline-block;width:10px;height:10px;background:#5f5;opacity:0.8;margin:0 6px 0 12px;vertical-align:middle'></span>B only
              <span style='display:inline-block;width:10px;height:10px;background:#ff6;opacity:0.8;margin:0 6px 0 12px;vertical-align:middle'></span>Both
            </div>"""

            # табличка метрик
            rows = [
                ("IoU", st["IoU"]), ("F1", st["F1"]),
                ("precision", st["precision"]), ("recall", st["recall"]),
                ("Area A (km²)", st["area_A_m2"]/1e6),
                ("Area B (km²)", st["area_B_m2"]/1e6),
                ("Δ|A−B| (km²)", st["delta_area_m2"]/1e6),
            ]
            stats_tbl = html.Table(
                [html.Tr([html.Th(k), html.Td(f"{v:.3f}" if isinstance(v, float) and np.isfinite(v) else v)]) for k, v in rows],
                style={"background":"#181818","color":"#eee","padding":"6px"}
            )

            # deck.gl spec
            spec = build_spec(build_dem_url("terrain"), diff_bitmap, basin_json)
            spec_obj = json.loads(spec)
            spec_obj.setdefault("description", {})["top-right"] = legend_html
            return json.dumps(spec_obj), fig, stats_tbl

        except Exception as e:
            logger.exception("Flood compare error: %s", e)
            return no_update, no_update, f"Flood comparison error: {e}"

    # --- dH-порівняння (continuous)
    p1 = _find_path_for(dem1, cat)
    p2 = _find_path_for(dem2, cat)
    try:
        diff, ref = compute_dem_difference(p1, p2)
    except Exception as e:
        logger.exception("Diff error: %s", e)
        return no_update, no_update, f"Computation error: {e}"

    # діапазон відображення
    try:
        if (cat or "").lower() == "dem":
            vmin, vmax = -25.0, 25.0
        else:
            q1, q99 = np.nanpercentile(diff, [1, 99])
            vmin, vmax = float(q1), float(q99)
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin, vmax = -10.0, 10.0
    except Exception:
        vmin, vmax = -10.0, 10.0

    # overlay
    try:
        img_uri = diff_to_base64_png(diff, ref, vmin=vmin, vmax=vmax, figsize=(8, 8))
        bounds = raster_bounds_ll(ref)
        diff_bitmap = bitmap_layer("diff-bitmap", img_uri, bounds)
    except Exception as e:
        logger.exception("Overlay build error: %s", e)
        return no_update, no_update, f"Rendering error (overlay): {e}"

    # легенда
    legend_uri = make_colorbar_datauri(vmin, vmax, cmap="RdBu_r")
    legend_html = f"<img src='{legend_uri}' style='height:160px'/>"

    # гістограма (Plotly) + таблиця статистик
    hist_fig = plotly_histogram_figure(diff, bins=60, clip_range=(vmin, vmax),
                                       density=False, cumulative=False)
    stats = calculate_error_statistics(diff)
    rows = [html.Tr([html.Th(k), html.Td(f"{v:.3f}" if isinstance(v, float) and np.isfinite(v) else v)])
            for k, v in stats.items()]
    stats_tbl = html.Table(rows, style={"background": "#181818", "color": "#eee", "padding": "6px"})

    # deck.gl spec
    spec = build_spec(build_dem_url("terrain"), diff_bitmap, basin_json)
    spec_obj = json.loads(spec)
    spec_obj.setdefault("description", {})["top-right"] = legend_html
    return json.dumps(spec_obj), hist_fig, stats_tbl


@app.callback(Output("deck-events","children"), Input("deck-main","lastEvent"))
def show_evt(evt):
    if not evt:
        return ""
    return f"{evt.get('eventType')} @ {evt.get('coordinate')}"
