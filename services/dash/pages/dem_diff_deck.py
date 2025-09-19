# pages/dem_diff_deck.py
import os
import json
import logging
from collections import defaultdict

import numpy as np
import geopandas as gpd

import dash
from dash import html, dcc, callback, Output, Input, State, no_update
import dash_deckgl

from utils.dem_tools import (
    compute_dem_difference,
    save_temp_diff_as_cog,
    make_colorbar_datauri,
    plot_histogram,
    calculate_error_statistics,
)
from registry import get_df


# ----------------------------
# Logging
# ----------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=os.getenv("PAGE_LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
logger.info("=== dem_diff_deck page init ===")


# ----------------------------
# ENV / constants
# ----------------------------
TC_BASE = os.getenv("TERRACOTTA_PUBLIC_URL", "https://www.geohydroai.org/tc").rstrip("/")
MAPBOX_ACCESS_TOKEN = os.getenv("MAPBOX_ACCESS_TOKEN", "").strip()
if not MAPBOX_ACCESS_TOKEN:
    logger.warning("MAPBOX_ACCESS_TOKEN is empty — DeckGL will fall back to Positron basemap.")

dash.register_page(__name__, path="/dem-diff", name="DEM Diff (deck.gl)", order=2)


# ----------------------------
# Basin (outline)
# ----------------------------
try:
    basin: gpd.GeoDataFrame = get_df("basin")
    logger.info("Basin loaded, CRS=%s, rows=%d", basin.crs, len(basin))
    basin = basin.to_crs("EPSG:4326")
    basin_json = json.loads(basin.to_json())
except Exception as e:
    logger.exception("Failed to load basin: %s", e)
    basin_json = None


# ----------------------------
# Layers index (from assets) + path normalization
# ----------------------------
ASSETS_INDEX_PATH = "assets/layers_index.json"
if not os.path.exists(ASSETS_INDEX_PATH):
    logger.warning("assets/layers_index.json not found at %s", ASSETS_INDEX_PATH)

def _fix_path(p: str) -> str:
    """Normalize legacy paths to container paths (/app/data/cogs/...)."""
    if not p:
        return p
    p = p.replace("\\", "/")
    p = p.replace("data/COG/", "/app/data/cogs/").replace("data/cogs/", "/app/data/cogs/")
    if p.startswith("data/"):
        p = "/app/" + p  # final safety net for any 'data/...'
    return p

try:
    with open(ASSETS_INDEX_PATH, "r") as f:
        raw_index = json.load(f)
    layers_index = []
    for rec in raw_index:
        r = dict(rec)
        if "path" in r and r["path"]:
            r["path"] = _fix_path(r["path"])
        layers_index.append(r)
    logger.info("Layers index loaded: %d entries", len(layers_index))
except Exception as e:
    logger.exception("Failed to load layers_index.json: %s", e)
    layers_index = []


# Group for filters
by_dem = defaultdict(list)
categories = set()
for l in layers_index:
    by_dem[l.get("dem")].append(l)
    if l.get("category"):
        categories.add(l["category"])

DEM_LIST = sorted([d for d in by_dem.keys() if d])
CATEGORY_LIST = sorted(categories)

logger.info("DEM_LIST=%s", DEM_LIST)
logger.info("CATEGORY_LIST=%s", CATEGORY_LIST)


# ----------------------------
# Deck.gl builders
# ----------------------------
def build_dem_url(colormap="viridis") -> str:
    """Base DEM for context (not diff)."""
    url = f"{TC_BASE}/singleband/dem/fab_dem/{{z}}/{{x}}/{{y}}.png?colormap={colormap}&stretch_range=[250,2200]"
    logger.debug("DEM base URL: %s", url)
    return url

def build_tempdiff_url(basename: str, vmin: float, vmax: float, cmap="RdBu_r") -> str:
    url = f"{TC_BASE}/tiles/{basename}/{{z}}/{{x}}/{{y}}.png?colormap={cmap}&stretch_range=[{vmin:.3f},{vmax:.3f}]"
    logger.debug("Diff tiles URL: %s", url)
    return url

def tile_layer(layer_id: str, url: str, opacity: float = 1.0) -> dict:
    return {
        "@@type": "TileLayer",
        "id": layer_id,
        "data": url,
        "minZoom": 0,
        "maxZoom": 19,
        "tileSize": 256,
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

def build_spec(
    dem_url: str | None,
    diff_url: str | None,
    basin: dict | None,
    init_view=None,
    map_style: str = "mapbox://styles/mapbox/dark-v11",
) -> str:
    layers = []
    if dem_url:
        layers.append(tile_layer("dem-tiles", dem_url, opacity=0.75))
    if diff_url:
        layers.append(tile_layer("diff-tiles", diff_url, opacity=0.95))
    if basin:
        layers.append(basin_layer(basin))
    spec = {
        "mapStyle": map_style,
        "initialViewState": init_view
        or {"longitude": 25.03, "latitude": 47.8, "zoom": 10, "pitch": 0, "bearing": 0},
        "layers": layers,
    }
    return json.dumps(spec)


# ----------------------------
# UI
# ----------------------------
layout = html.Div(
    [
        html.H3("DEM Difference Analysis (deck.gl + Terracotta)"),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("DEM 1"),
                        dcc.Dropdown(
                            id="dem1",
                            options=[{"label": d, "value": d} for d in DEM_LIST],
                        ),
                        html.Label("DEM 2"),
                        dcc.Dropdown(
                            id="dem2",
                            options=[{"label": d, "value": d} for d in DEM_LIST],
                        ),
                        html.Label("Категорія"),
                        dcc.Dropdown(
                            id="cat",
                            options=[{"label": c, "value": c} for c in CATEGORY_LIST],
                        ),
                        html.Br(),
                        html.Button("Порахувати різницю", id="run"),
                    ],
                    style={
                        "width": "340px",
                        "marginRight": "16px",
                        "display": "inline-block",
                        "verticalAlign": "top",
                    },
                ),
                html.Div(
                    [
                        dash_deckgl.DashDeckgl(
                            id="deck-main",
                            spec=build_spec(build_dem_url("viridis"), None, basin_json),
                            description={"top-right": "<div id='legend'>Legend</div>"},
                            height=640,
                            cursor_position="bottom-right",
                            events=["hover"],
                            mapbox_key=MAPBOX_ACCESS_TOKEN,
                        ),
                        html.Div(
                            id="deck-events",
                            style={"fontFamily": "monospace", "marginTop": "6px"},
                        ),
                    ],
                    style={
                        "width": "calc(100% - 360px)",
                        "display": "inline-block",
                        "verticalAlign": "top",
                    },
                ),
            ]
        ),
        html.Div(
            [
                html.Div(
                    [html.H4("Гістограма"), html.Img(id="hist", style={"height": "220px"})],
                    style={"display": "inline-block", "marginRight": "24px"},
                ),
                html.Div(
                    id="stats",
                    style={
                        "display": "inline-block",
                        "verticalAlign": "top",
                        "fontFamily": "monospace",
                    },
                ),
            ],
            style={"marginTop": "14px"},
        ),
        html.Hr(),
        html.H4("Режим порівняння (дві панелі)"),
        html.Div(
            [
                dash_deckgl.DashDeckgl(
                    id="deck-left",
                    spec=build_spec(build_dem_url("terrain"), None, basin_json),
                    height=420,
                    cursor_position="none",
                    mapbox_key=MAPBOX_ACCESS_TOKEN,
                ),
                dash_deckgl.DashDeckgl(
                    id="deck-right",
                    spec=build_spec(build_dem_url("viridis"), None, basin_json),
                    height=420,
                    cursor_position="none",
                    mapbox_key=MAPBOX_ACCESS_TOKEN,
                ),
            ],
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "8px"},
        ),
    ]
)


# ----------------------------
# Helpers
# ----------------------------
def _pick_path(name: str, category: str | None) -> str | None:
    arr = by_dem.get(name, [])
    if not arr:
        return None
    if category:
        for it in arr:
            if it.get("category") == category:
                return it.get("path")
    return arr[0].get("path")


# ----------------------------
# Callbacks
# ----------------------------
@callback(
    Output("deck-main", "spec"),
    Output("hist", "src"),
    Output("stats", "children"),
    Input("run", "n_clicks"),
    State("dem1", "value"),
    State("dem2", "value"),
    State("cat", "value"),
    prevent_initial_call=True,
)
def run_diff(n, dem1, dem2, cat):
    logger.info("Run clicked: n=%s, dem1=%s, dem2=%s, cat=%s", n, dem1, dem2, cat)

    if not dem1 or not dem2 or dem1 == dem2:
        logger.warning("Invalid DEM selection")
        return no_update, no_update, "Оберіть різні DEM!"

    p1, p2 = _pick_path(dem1, cat), _pick_path(dem2, cat)
    if not p1 or not p2:
        logger.error("DEM not found in layers_index: p1=%s, p2=%s", p1, p2)
        return no_update, no_update, "DEM не знайдено у layers_index!"

    # дубль нормалізації (на випадок майбутніх оновлень індексу)
    p1, p2 = _fix_path(p1), _fix_path(p2)
    logger.info("Paths: dem1=%s | dem2=%s", p1, p2)

    # 1) compute diff
    try:
        diff, ref = compute_dem_difference(p1, p2)
        logger.info("Diff computed: shape=%s, nan%%=%.2f",
                    diff.shape, float(np.isnan(diff).mean() * 100.0))
    except Exception as e:
        logger.exception("Error computing diff: %s", e)
        return no_update, no_update, f"Помилка при обчисленні: {e}"

    # 2) dynamic stretch
    try:
        q1, q99 = np.nanpercentile(diff, [1, 99])
        vmin, vmax = float(q1), float(q99)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = -10.0, 10.0
    except Exception:
        vmin, vmax = -10.0, 10.0
    logger.info("Stretch range: vmin=%.3f, vmax=%.3f", vmin, vmax)

    # 3) save temp COG in /tmp and build tiles URL
    try:
        cog_path = save_temp_diff_as_cog(diff, ref, prefix="demdiff_")
        basename = os.path.basename(cog_path)
        diff_url = build_tempdiff_url(basename, vmin, vmax, cmap="RdBu_r")
        logger.info("Temp COG saved: %s (basename=%s)", cog_path, basename)
        logger.info("Diff tiles URL: %s", diff_url)
    except Exception as e:
        logger.exception("Error saving COG or building URL: %s", e)
        return no_update, no_update, f"Помилка збереження COG/URL: {e}"

    # 4) legend, histogram, stats
    legend_uri = make_colorbar_datauri(vmin, vmax, cmap="RdBu_r")
    legend_html = f"<img src='{legend_uri}' style='height:160px'/>"

    hist = plot_histogram(diff, clip_range=(vmin, vmax))
    stats = calculate_error_statistics(diff)
    logger.info(
        "Stats: count=%s mean=%.3f median=%.3f std=%.3f rmse=%.3f min=%.3f max=%.3f",
        stats.get("count"),
        float(stats.get("mean_error") or np.nan),
        float(stats.get("median_error") or np.nan),
        float(stats.get("std_dev") or np.nan),
        float(stats.get("rmse") or np.nan),
        float(stats.get("min") or np.nan),
        float(stats.get("max") or np.nan),
    )

    rows = [
        html.Tr(
            [
                html.Th(k),
                html.Td(f"{v:.3f}" if isinstance(v, float) and np.isfinite(v) else v),
            ]
        )
        for k, v in stats.items()
    ]
    stats_tbl = html.Table(
        rows, style={"background": "#181818", "color": "#eee", "padding": "6px"}
    )

    # 5) final spec with legend on the map (top-right)
    spec = build_spec(build_dem_url("terrain"), diff_url, basin_json)
    spec_obj = json.loads(spec)
    spec_obj.setdefault("description", {})["top-right"] = legend_html

    return json.dumps(spec_obj), hist, stats_tbl


@callback(Output("deck-events", "children"), Input("deck-main", "lastEvent"))
def show_evt(evt):
    if not evt:
        return ""
    return f"{evt.get('eventType')} @ {evt.get('coordinate')}"
