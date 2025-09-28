# pages/dem_diff_deck.py
import os
import json
import logging
from collections import defaultdict

import numpy as np
import geopandas as gpd
from xdem import DEM as _DEM

import dash
from dash import html, dcc, callback, Output, Input, State, no_update
import dash_deckgl

from registry import get_df
from utils.style import empty_dark_figure
from utils.dem_tools import (
    compute_dem_difference,
    make_colorbar_datauri,
    calculate_error_statistics,
    diff_to_base64_png,
    raster_bounds_ll,
    plotly_histogram_figure,
    plotly_violin_figure,
    plotly_ecdf_figure,
    read_binary_with_meta,
    align_boolean_pair,
    crop_to_common_extent,
    pixel_area_m2_from_ref_dem,
    flood_metrics,
    plotly_flood_areas_figure,
    flood_compare_overlay_png,
    robust_stats,
)

# ---------- Константи UI ----------

# ---------- Константи UI ----------
MAIN_MAP_HEIGHT = 480
RIGHT_PANEL_WIDTH = 500
MAP_WIDTH_PX = 340           # ← максимальна ширина карти (пікселі)
ZOOM_DEFAULT = 12            # ← бажаний стартовий зум

# ---------- Логи ----------

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=os.getenv("PAGE_LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
logger.info("=== dem_diff_deck page init ===")

# ---------- Зовнішні сервіси ----------

TC_BASE = os.getenv("TERRACOTTA_PUBLIC_URL", "https://www.geohydroai.org/tc").rstrip("/")
MAPBOX_ACCESS_TOKEN = os.getenv("MAPBOX_ACCESS_TOKEN", "").strip()

dash.register_page(__name__, path="/dem-diff", name="DEM Diff", order=2)
app = dash.get_app()

# ---------- Basin (опційно) ----------

try:
    basin: gpd.GeoDataFrame = get_df("basin")
    logger.info("Basin loaded, CRS=%s, rows=%d", basin.crs, len(basin))
    basin = basin.to_crs("EPSG:4326")
    basin_json = json.loads(basin.to_json())
except Exception as e:
    logger.exception("Failed to load basin: %s", e)
    basin_json = None

# ---------- layers_index.json ----------

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

# ---------- Групування за DEM та категоріями ----------

by_dem = defaultdict(list)
categories = set()
for l in layers_index:
    by_dem[l.get("dem")].append(l)
    if l.get("category"):
        categories.add(l["category"])

DEM_LIST = sorted([d for d in by_dem.keys() if d])
CATEGORY_LIST = sorted(categories)


# ---------- Flood index (допоміжне) ----------

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
    return (
        idx,
        sorted(dem_values),
        sorted(hand_values),
        sorted(flood_values, key=_flood_level_key),
    )


FLOOD_INDEX, FLOOD_DEMS, FLOOD_HANDS, FLOOD_LEVELS = build_flood_index(layers_index)


# ---------- deck.gl helpers ----------

def tile_layer(layer_id: str, url: str, opacity: float = 1.0) -> dict:
    return {
        "@@type": "TileLayer",
        "id": layer_id,
        "data": url,
        "minZoom": 0,
        "maxZoom": 19,
        "tileSize": 220,
        "opacity": opacity,
        "renderSubLayers": {
            "@@function": [
                "tile",
                {
                    "type": "BitmapLayer",
                    "id": f"{layer_id}-bitmap",
                    "image": "@@tile.data",
                    "bounds": "@@tile.bbox",
                    "opacity": opacity,
                },
            ]
        },
    }

def bitmap_layer(layer_id: str, image_data_uri: str, bounds) -> dict:
    # bounds: [[S,W],[N,E]] → [W,S,E,N]
    (south, west), (north, east) = bounds
    return {
        "@@type": "BitmapLayer",
        "id": layer_id,
        "image": image_data_uri,
        "bounds": [west, south, east, north],
        "opacity": 0.95,
    }


def basin_layer(geojson: dict) -> dict:
    return {
        "@@type": "GeoJsonLayer",
        "id": "basin-outline",
        "data": geojson,
        "stroked": True,
        "filled": False,
        "getLineColor": [0, 102, 255, 200],
        "getLineWidth": 2,
        "lineWidthUnits": "pixels",
    }


def build_dem_url(colormap="viridis"):
    return f"{TC_BASE}/singleband/dem/fab_dem/{{z}}/{{x}}/{{y}}.png?colormap={colormap}&stretch_range=[250,2200]"

def view_from_geojson_bbox(gj, fallback=None, zoom_default=10):
    try:
        xs, ys = [], []
        for feat in gj.get("features", []):
            def _walk(c):
                if isinstance(c[0], (float, int)):
                    xs.append(c[0]); ys.append(c[1])
                else:
                    for cc in c: _walk(cc)
            _walk(feat["geometry"]["coordinates"])
        west, east = min(xs), max(xs)
        south, north = min(ys), max(ys)
        lon = (west + east) / 2.0
        lat = (south + north) / 2.0
        # фіксуємо зум (можна легко змінити)
        return {"longitude": float(lon), "latitude": float(lat),
                "zoom": float(zoom_default), "pitch": 0, "bearing": 0}
    except Exception:
        return fallback or {"longitude": 25.03, "latitude": 47.8, "zoom": float(zoom_default)}

def build_spec(dem_url, diff_bitmap, basin, init_view=None,
               map_style="mapbox://styles/mapbox/light-v11"):
    layers = []
    if dem_url: layers.append(tile_layer("dem-tiles", dem_url, opacity=0.85))
    if diff_bitmap: layers.append(diff_bitmap)
    if basin: layers.append(basin_layer(basin))

    init_vs = init_view or (
        view_from_geojson_bbox(basin, fallback={"longitude":25.03,"latitude":47.8,"zoom":ZOOM_DEFAULT},
                               zoom_default=ZOOM_DEFAULT) if basin
        else {"longitude":25.03,"latitude":47.8,"zoom":ZOOM_DEFAULT}
    )

    return json.dumps({
        "mapStyle": map_style,
        "initialViewState": init_vs,
        "controller": True,
        "layers": layers,
    })


# ---------- Layout (ФІНАЛЬНИЙ ОНОВЛЕНИЙ) ----------

layout =  html.Div(
    [
    html.Div(
    [
        html.H3("DEM Difference Analysis"),
        html.Div(
            [
                # Ліва панель (Controls)
                html.Div(
                    [
                        dcc.Dropdown(
                            id="dem1",
                            options=[{"label": d, "value": d} for d in DEM_LIST],
                            value=(
                                "copernicus_dem"
                                if "copernicus_dem" in DEM_LIST
                                else (DEM_LIST[0] if DEM_LIST else None)
                            ),
                            style={"marginBottom": "10px", "fontSize": "14px"},
                        ),
                        dcc.Dropdown(
                            id="dem2",
                            options=[{"label": d, "value": d} for d in DEM_LIST],
                            value=(
                                "srtm_dem"
                                if "srtm_dem" in DEM_LIST
                                else (DEM_LIST[1] if len(DEM_LIST) > 1 else None)
                            ),
                            style={"marginBottom": "10px", "fontSize": "14px"},
                        ),
                        dcc.Dropdown(
                            id="cat",
                            options=[{"label": c, "value": c} for c in CATEGORY_LIST],
                            value=(
                                "dem"
                                if "dem" in CATEGORY_LIST
                                else (CATEGORY_LIST[0] if CATEGORY_LIST else None)
                            ),
                            style={"marginBottom": "8px", "fontSize": "14px"},
                        ),
                        # Параметри тільки для flood_scenarios
                        html.Div(
                            [
                                html.Label(
                                    "HAND model", style={"marginBottom": "4px"}
                                ),
                                dcc.Dropdown(
                                    id="flood_hand",
                                    options=[
                                        {"label": h, "value": h} for h in FLOOD_HANDS
                                    ],
                                    value=(FLOOD_HANDS[0] if FLOOD_HANDS else None),
                                    style={"marginBottom": "8px"},
                                ),
                                html.Label(
                                    "Flood level", style={"marginBottom": "4px"}
                                ),
                                dcc.Dropdown(
                                    id="flood_level",
                                    options=[
                                        {"label": f, "value": f} for f in FLOOD_LEVELS
                                    ],
                                    value=(FLOOD_LEVELS[0] if FLOOD_LEVELS else None),
                                ),
                            ],
                            id="flood_opts",
                            style={"display": "none", "marginBottom": "10px"},
                        ),
                        html.Button(
                            "Compute Difference",
                            id="run",
                            style={
                                "backgroundColor": "#1f77b4",
                                "color": "white",
                                "border": "none",
                                "borderRadius": "6px",
                                "padding": "6px 12px",
                                "cursor": "pointer",
                                "fontWeight": "bold",
                                "fontSize": "14px",
                                "width": "100%",
                            },
                        ),
                    ],
                    style={
                        "border": "1px solid rgba(255,255,255,0.15)",
                        "width": "180px",
                        "padding": "12px",
                        "backgroundColor": "#1e1e1e",
                        "borderRadius": "8px",
                    },
                ),
                # Права зона (Карта + Гістограма)
                html.Div(
                    [
                        # Карта (ТЕПЕР БЕЗ ОВЕРЛЕЮ ЛЕГЕНДИ)
                        html.Div(
                            [
                                dash_deckgl.DashDeckgl(
                                    id="deck-main",
                                    spec=build_spec(build_dem_url("viridis"), None, basin_json),
                                    height=MAIN_MAP_HEIGHT,
                                    cursor_position="bottom-right",
                                    events=["hover"],
                                    mapbox_key=MAPBOX_ACCESS_TOKEN,
                                ),
                                # !!! html.Div(id="legend-box") ВИДАЛЕНО ЗВІДСИ
                            ],
                            style={
                                # !!! position: relative ВИДАЛЕНО
                                "border": "1px solid rgba(255,255,255,0.15)",
                                "borderRadius": "8px",
                                "overflow": "hidden",
                                "boxShadow": "0 4px 16px rgba(0,0,0,0.3)",
                                "backgroundColor": "#111",
                                "width": f"{MAP_WIDTH_PX}px",  # ← щоб не розтягувалась
                            },
                        ),

                        # Права панель (Гістограма + Легенда)
                        # Права панель (графіки + легенда)
                        html.Div(
                            [
                                dcc.Graph(
                                    id="hist",
                                    figure=empty_dark_figure(220, "Press “Compute Difference”"),
                                    style={"height": "200px"},
                                    config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
                                ),
                                dcc.Graph(
                                    id="violin",
                                    figure=empty_dark_figure(180, "Violin will appear after run"),
                                    style={"height": "170px", "marginTop": "6px"},
                                    config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
                                ),
                                dcc.Graph(
                                    id="ecdf",
                                    figure=empty_dark_figure(180, "ECDF will appear after run"),
                                    style={"height": "170px", "marginTop": "6px"},
                                    config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
                                ),
                                html.Hr(style={"borderColor": "rgba(255,255,255,0.15)"}),
                                html.Div(
                                    id="legend-box",
                                    style={
                                        "padding": "8px 10px",
                                        "fontFamily": "monospace",
                                        "fontSize": "12px",
                                        "background": "#1e1e1e",
                                    },
                                ),
                            ],
                            style={
                                "width": f"{RIGHT_PANEL_WIDTH}px",
                                "maxWidth": f"{RIGHT_PANEL_WIDTH}px",
                                "paddingLeft": "10px",
                                "overflowY": "auto",
                                "height": f"{MAIN_MAP_HEIGHT}px",
                            },
                        ),

                    ],
                    style={
                        "display": "grid",
                        "gridTemplateColumns": f"{MAP_WIDTH_PX}px {RIGHT_PANEL_WIDTH}px",  # ← фіксована ширина карти
                        "gap": "10px",
                        "alignItems": "start",
                        "gridColumn": "2 / 3",
                    },
                ),

                # НОВИЙ БЛОК: Статистика під картою
                html.Div(
                    [
                        html.H4("Analysis Results", style={"marginTop": "15px", "marginBottom": "5px"}),
                        html.Div(
                            id="stats",
                            style={
                                "fontFamily": "monospace",
                                "backgroundColor": "#1e1e1e",
                                "padding": "15px",
                                "borderRadius": "8px",
                                "border": "1px solid rgba(255,255,255,0.15)",
                            }
                        ),
                    ],
                    style={
                        "gridColumn": "2 / 3",
                        "marginTop": "10px"
                    }
                ),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "230px 1fr",
                "gap": "10px",
                "alignItems": "start",
                "gridTemplateRows": "auto auto"
            },
        ),
        html.Div(
            id="deck-events",
            style={
                "fontFamily": "monospace",
                "marginTop": "16px",
                "padding": "8px",
                "backgroundColor": "#1e1e1e",
                "color": "#eee",
                "borderRadius": "4px",
                "border": "1px solid rgba(255,255,255,0.15)",
            }
        ),
    ],
        id="page-container",
        style={
            "maxWidth": "1420px",  # ← обмежуємо загальну ширину сторінки
            "margin": "0 auto",  # ← центруємо по горизонталі
            "padding": "0 16px"  # ← невеликий внутрішній відступ, щоб контент не лип до країв
        },
    )
    ],
    style={"width": "100%"}
)
# ---------- Службові (TOOLTIPS/СТИЛІ) ----------

TOOLTIP_TEXTS = {
    # Basic Stats
    "count": "Number of valid pixels used for calculation.",
    "mean_error": "Mean Error. Arithmetic average of dH (DEM2 - DEM1). Highly sensitive to outliers.",
    "median_error": "Median Error. The middle value of dH. Robust estimate of systematic bias (same as Robust Median).",
    "std_dev": "Standard Deviation (STD). Measure of dispersion around the mean. Highly sensitive to outliers.",
    "rmse": "Root Mean Square Error (RMSE). Emphasizes large errors (squared difference). Useful to compare with MAE.",
    "min": "Minimum value of the difference dH (DEM2 - DEM1).",
    "max": "Maximum value of the difference dH (DEM2 - DEM1).",
    "skewness": "Skewness. Asymmetry of the distribution. Indicates if the tail is longer on one side (not robust).",
    "kurtosis": "Kurtosis. 'Tailedness' or sharpness of the distribution peak (not robust).",

    # Robust Stats
    "median": "Median dH. Robust estimate of systematic bias (shift) between DEMs.",
    "nmad": "Normalized Median Absolute Deviation (NMAD). Robust analogue of STD. Represents the 'typical random error' per pixel.",
    "mae": "Mean Absolute Error (MAE). Average magnitude of the difference |dH|. Less sensitive to outliers.",
    "p5": "5th Percentile (P5). Lower bound of the central 90% of differences.",
    "p95": "95th Percentile (P95). Upper bound of the central 90% of differences.",
    "outlier_%": "Percentage of outliers where |dH - median| > 3 * NMAD. Indicator of data inconsistency/noise.",
}


# Стилі для підсвічування
WARNING_STYLE = {"color": "gold", "fontWeight": "bold"}
CRITICAL_STYLE = {"color": "#ff6666", "fontWeight": "bold"}
DEFAULT_STYLE = {"color": "#eee"}


def _get_style(key, value, basic, robust):
    """Повертає стиль на основі значення та порівнянь."""
    if not isinstance(value, (int, float)) or not np.isfinite(value):
        return DEFAULT_STYLE

    # Критична логіка для робастних метрик
    if key == "outlier_%":
        if value > 5.0:
            return CRITICAL_STYLE
        if value > 2.0:
            return WARNING_STYLE

    # Систематичний зсув
    if key in ["median", "mean_error"]:
        val_to_check = value if key == "mean_error" else robust.get("median", value)
        if abs(val_to_check) > 5.0:
            return WARNING_STYLE

    # RMSE/MAE порівняння (вказує на наявність великих викидів)
    if key in ["rmse", "mae"]:
        rmse = robust.get("rmse", np.nan)
        mae = robust.get("mae", np.nan)
        if np.isfinite(rmse) and np.isfinite(mae) and (rmse / mae) > 2.0:
            return WARNING_STYLE

    return DEFAULT_STYLE


def _pick_path(name, category):
    arr = by_dem.get(name, [])
    if not arr:
        return None
    if category:
        for it in arr:
            if it.get("category") == category:
                return it.get("path")
    return arr[0].get("path")


@callback(Output("flood_opts", "style"), Input("cat", "value"))
def _toggle_flood_opts(cat):
    return (
        {"display": "block", "marginBottom": "10px"}
        if cat == "flood_scenarios"
        else {"display": "none"}
    )


def _flood_stats_table(st: dict):
    rows = [
        ("IoU", st["IoU"]),
        ("F1", st["F1"]),
        ("precision", st["precision"]),
        ("recall", st["recall"]),
        ("Area A (km²)", st["area_A_m2"] / 1e6),
        ("Area B (km²)", st["area_B_m2"] / 1e6),
        ("Δ|A−B| (km²)", st["delta_area_m2"] / 1e6),
    ]
    return html.Table(
        [
            html.Tr(
                [
                    html.Th(k),
                    html.Td(f"{v:.3f}" if isinstance(v, float) and np.isfinite(v) else v),
                ]
            )
            for k, v in rows
        ],
        style={"background": "#181818", "color": "#eee", "padding": "6px", "width": "100%"},
    )


def _dh_stats_tables(basic: dict, robust: dict):
    def _row(k, v, source_dict):
        style = _get_style(k, v, basic, robust)
        tooltip = TOOLTIP_TEXTS.get(k, k)
        is_num = isinstance(v, (int, float)) and np.isfinite(v)

        return html.Tr(
            [
                html.Th(
                    k,
                    title=tooltip,
                    style={"cursor": "help", **DEFAULT_STYLE, "paddingRight": "15px", "textAlign": "left"}
                ),
                html.Td(
                    f"{v:.3f}" if is_num else v,
                    style=style
                ),
            ]
        )

    basic_keys = ["count", "mean_error", "median_error", "std_dev", "rmse", "min", "max", "skewness", "kurtosis"]
    robust_keys = ["median", "nmad", "mae", "rmse", "p5", "p95", "outlier_%", "skew", "kurtosis"]

    return html.Div(
        style={"display": "flex", "gap": "30px", "justifyContent": "flex-start"},
        children=[
            # Блок Basic Stats
            html.Div(
                style={"flex": "1 1 50%", "minWidth": "300px"},
                children=[
                    html.Div(
                        "Basic Statistics (Standard)",
                        style={"fontWeight": "bold", "margin": "0 0 4px", "color": "#ccc", "fontSize": "16px"}
                    ),
                    html.Table(
                        [_row(k, basic.get(k, np.nan), basic) for k in basic_keys],
                        style={"width": "100%", "borderCollapse": "collapse"},
                    ),
                ]
            ),
            # Блок Robust Stats
            html.Div(
                style={"flex": "1 1 50%", "minWidth": "300px"},
                children=[
                    html.Div(
                        "Robust Statistics (Outlier-Resistant)",
                        style={"fontWeight": "bold", "margin": "0 0 4px", "color": "#ccc", "fontSize": "16px"}
                    ),
                    html.Table(
                        [_row(k, robust.get(k, np.nan), robust) for k in robust_keys],
                        style={"width": "100%", "borderCollapse": "collapse"},
                    ),
                ]
            ),
        ]
    )


@app.callback(Output("deck-events", "children"), Input("deck-main", "lastEvent"))
def show_evt(evt):
    if not evt:
        return ""
    return f"{evt.get('eventType')} @ {evt.get('coordinate')}"



# ---------- Основний колбек (ФІНАЛЬНИЙ ОНОВЛЕНИЙ) ----------

@app.callback(
    Output("deck-main", "spec"),
    Output("hist", "figure"),
    Output("stats", "children"),
    Output("violin", "figure"),
    Output("ecdf", "figure"),
    Output("legend-box", "children"),  # <-- OUTPUT ДЛЯ ЛЕГЕНДИ
    Input("run", "n_clicks"),
    State("dem1", "value"),
    State("dem2", "value"),
    State("cat", "value"),
    State("flood_hand", "value"),
    State("flood_level", "value"),
    prevent_initial_call=False,
)
def run_diff(n, dem1, dem2, cat, flood_hand, flood_level):
    # Нормалізуємо дефолти спочатку
    default_dem1 = "copernicus_dem" if "copernicus_dem" in DEM_LIST else (DEM_LIST[0] if DEM_LIST else None)
    default_dem2 = "srtm_dem" if "srtm_dem" in DEM_LIST else (DEM_LIST[1] if len(DEM_LIST) > 1 else None)
    default_cat = "dem" if "dem" in CATEGORY_LIST else (CATEGORY_LIST[0] if CATEGORY_LIST else None)

    dem1 = dem1 or default_dem1
    dem2 = dem2 or default_dem2
    cat = cat or default_cat

    # Початковий стан легенди (коли не обчислено)
    initial_legend_content = html.Div([
        html.B("DEM tiles"),
        html.Div("Оберіть DEM1/DEM2 і натисніть “Compute Difference”.")
    ])

    if not dem1 or not dem2:
        return no_update, no_update, no_update, no_update, "No DEMs available.", initial_legend_content
    if dem1 == dem2 and len(DEM_LIST) > 1:
        dem2 = next((d for d in DEM_LIST if d != dem1), dem2)

    # Функція пошуку шляху (скопійована з вашого коду)
    def _find_path_for(dem_name, category, hand=None, level=None):
        cand = [r for r in by_dem.get(dem_name, []) if r.get("category") == category]
        if category == "flood_scenarios":
            if hand:
                cand = [r for r in cand if r.get("hand") == hand]
            if level:
                cand = [r for r in cand if r.get("flood") == level]
            if not cand:
                return None
            cand.sort(
                key=lambda r: (r.get("hand") is not None, r.get("flood") is not None),
                reverse=True,
            )
            return _fix_path(cand[0]["path"])
        return _fix_path(cand[0]["path"]) if cand else None

    # ---------- FLOOD сценарії (Бінарне порівняння) ----------
    if cat == "flood_scenarios":
        p1 = _find_path_for(dem1, "flood_scenarios", flood_hand, flood_level)
        p2 = _find_path_for(dem2, "flood_scenarios", flood_hand, flood_level)

        if not p1 or not p2:
            return no_update, no_update, "Flood layer not found for selected DEM/HAND/level.", initial_legend_content

        try:
            A, A_tx, A_crs, A_w, A_h, _ = read_binary_with_meta(p1)
            B, B_tx, B_crs, B_w, B_h, _ = read_binary_with_meta(p2)
            A_aligned, B_aligned = align_boolean_pair(A, A_tx, A_crs, A_w, A_h, B, B_tx, B_crs, B_w, B_h)
            if A_aligned.shape != B_aligned.shape:
                A_aligned, B_aligned = crop_to_common_extent(A_aligned, B_aligned)

            base_dem_path = _pick_path(dem1, "dem") or _pick_path(dem1, None) or _pick_path(dem2, "dem") or p1
            ref = _DEM(_fix_path(base_dem_path))
            px_area = pixel_area_m2_from_ref_dem(ref)
            st = flood_metrics(A_aligned, B_aligned, px_area)

            fig = plotly_flood_areas_figure(st, title=f"Flood {flood_hand} @ {flood_level}: areas & Δ")

            overlay_uri = flood_compare_overlay_png(A_aligned, B_aligned, ref)
            bounds = raster_bounds_ll(ref)
            diff_bitmap = bitmap_layer("flood-diff-bitmap", overlay_uri, bounds)

            # Легенда FLOOD
            flood_legend_component = html.Div([
                html.Div("Flood Comparison", style={"fontWeight": 700, "marginBottom": "6px"}),
                html.Div([
                    html.Span(style={"display": "inline-block", "width": "10px", "height": "10px",
                                     "background": "rgba(255,0,0,0.7)", "marginRight": "6px",
                                     "verticalAlign": "middle", "borderRadius": "2px"}),
                    html.Span("A only", style={"color": "#ff6666"}),
                    html.Span("  (Lost area in DEM₂)",
                              style={"color": "#aaa", "fontSize": "11px", "marginLeft": "6px"}),
                ], style={"lineHeight": "1.4"}),
                html.Div([
                    html.Span(style={"display": "inline-block", "width": "10px", "height": "10px",
                                     "background": "rgba(0,255,0,0.7)", "marginRight": "6px",
                                     "verticalAlign": "middle", "borderRadius": "2px"}),
                    html.Span("B only", style={"color": "#66ff66"}),
                    html.Span("  (Gained area in DEM₂)",
                              style={"color": "#aaa", "fontSize": "11px", "marginLeft": "6px"}),
                ], style={"lineHeight": "1.4"}),
                html.Div([
                    html.Span(style={"display": "inline-block", "width": "10px", "height": "10px",
                                     "background": "rgba(255,255,0,0.7)", "marginRight": "6px",
                                     "verticalAlign": "middle", "borderRadius": "2px"}),
                    html.Span("Both", style={"color": "#ffff66"}),
                    html.Span("  (Consistent area)", style={"color": "#aaa", "fontSize": "11px", "marginLeft": "6px"}),
                ], style={"lineHeight": "1.4"}),
            ])

            spec = build_spec(build_dem_url("terrain"), diff_bitmap, basin_json)
            empty = empty_dark_figure(160, "")
            spec_obj = json.loads(spec)
            return json.dumps(spec_obj), fig, empty, empty, _flood_stats_table(st), html.Div(flood_legend_component)

        except Exception as e:
            logger.exception("Flood compare error: %s", e)
            return no_update, no_update, f"Flood comparison error: {e}", initial_legend_content

    # ---------- dH (continuous) (Порівняння DEM) ----------
    p1 = _find_path_for(dem1, cat)
    p2 = _find_path_for(dem2, cat)
    try:
        diff, ref = compute_dem_difference(p1, p2)
    except Exception as e:
        logger.exception("Diff error: %s", e)
        return no_update, no_update, no_update, no_update, f"Computation error: {e}", initial_legend_content

    # Діапазон відображення (vmin, vmax)
    try:
        # ВИКОРИСТОВУЄМО ДИНАМІЧНИЙ ДІАПАЗОН (1-99 перцентиль)
        q1, q99 = np.nanpercentile(diff, [1, 99])
        vmin, vmax = float(q1), float(q99)

        # Якщо перцентилі занадто вузькі або DEM, повертаємося до фіксованого/ширшого діапазону
        # (Це логіка для забезпечення візуального контрасту)
        if (cat or "").lower() == "dem" or (abs(vmax - vmin) < 5.0):  # Якщо діапазон менше 5м
            vmin, vmax = -25.0, 25.0
            # Якщо медіана сильно зміщена, центруємо діапазон навколо неї
            median = np.nanmedian(diff)
            if abs(median) > 5.0 and abs(median) < 20.0:
                vmin = median - 25.0
                vmax = median + 25.0

        # Фінальна перевірка на NaN
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = -10.0, 10.0
    except Exception:
        vmin, vmax = -10.0, 10.0

    # Overlay PNG
    try:
        # img_uri тепер створюється на основі vmin/vmax
        img_uri = diff_to_base64_png(diff, ref, vmin=vmin, vmax=vmax, figsize=(8, 8))
        bounds = raster_bounds_ll(ref)
        diff_bitmap = bitmap_layer("diff-bitmap", img_uri, bounds)
    except Exception as e:
        logger.exception("Overlay build error: %s", e)
        return no_update, no_update, no_update, no_update, f"Rendering error (overlay): {e}", initial_legend_content

    # Легенда: використовуємо отримані vmin, vmax, палітру RdBu_r та центр 0
    # Встановлюємо center=0, якщо 0 знаходиться в межах [vmin, vmax], інакше центр не потрібен
    center = 0.0 if (vmin < 0 and vmax > 0) else None
    legend_uri = make_colorbar_datauri(vmin, vmax, cmap="RdBu_r", label="ΔH (m)", center=center)

    # Створюємо HTML, який буде вставлений у legend-box
    legend_component = html.Div([
        html.Div(f"Elevation Difference (dH): {dem2} (Test) − {dem1} (Ref)",
                 title="dH = DEM_Test - DEM_Ref. Positive means DEM_Test is higher.",
                 style={"fontWeight": 700, "marginBottom": "4px", "fontSize": "12px", "cursor": "help"}),        html.Img(src=legend_uri, style={
            "height": "100px",  # було 160px
            "display": "block",
            "margin": "4px auto 2px",
            "border": "1px solid rgba(255,255,255,0.1)",
            "borderRadius": "6px"
        }),
        html.Div([
            html.Div([html.Span("• BLUE: + Change", style={"color": "#6699ff"})],
                     style={"lineHeight": "1.3", "fontSize": "11px"}),
            html.Div(f"{dem2} is HIGHER than {dem1} (Uplift/Bias)",
                     style={"marginLeft": "14px", "fontSize": "10px", "color": "#aaa"}),
            html.Div([html.Span("• RED: − Change", style={"color": "#ff6666"})],
                     style={"marginTop": "4px", "lineHeight": "1.3", "fontSize": "11px"}),
            html.Div(f"{dem2} is LOWER than {dem1} (Subsidence/Erosion)",
                     style={"marginLeft": "14px", "fontSize": "10px", "color": "#aaa"}),
            html.Hr(style={"borderColor": "rgba(255,255,255,0.1)", "margin": "6px 0"}),
            html.Div(f"Range: [{vmin:.2f}, {vmax:.2f}] m", style={"fontSize": "11px", "fontWeight": 700}),
        ], style={"textAlign": "left"})
    ], style={"padding": "6px 8px", "background": "#1e1e1e", "borderRadius": "8px"})

    clip = (vmin, vmax)
    # Гістограма
    hist_fig = plotly_histogram_figure(diff, bins=60, clip_range=clip, density=False, cumulative=False)
    violin_fig = plotly_violin_figure(diff, clip_range=clip, title="Distribution (Violin)")
    ecdf_fig = plotly_ecdf_figure(diff, clip_range=clip, title="ECDF of dH")
    # Статистика: Basic + Robust
    basic = calculate_error_statistics(diff)
    robust = robust_stats(diff, clip=(1, 99))
    stats_tbl = _dh_stats_tables(basic, robust)

    # deck.gl spec: додаємо diff_bitmap
    spec = build_spec(build_dem_url("terrain"), diff_bitmap, basin_json)

    spec_obj = json.loads(spec)

    # Повертаємо 4 значення
    return json.dumps(spec_obj), hist_fig, violin_fig, ecdf_fig, stats_tbl, html.Div(legend_component)
