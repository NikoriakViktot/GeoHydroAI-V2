# debug_utils.py
from __future__ import annotations
import time, json, logging, functools, traceback

log = logging.getLogger("debug")


def _to_jsonable(x):
    try:
        return json.loads(json.dumps(x, default=str))
    except Exception:
        return str(x)


def trace(name: str | None = None):
    def deco(fn):
        nm = name or fn.__name__
        @functools.wraps(fn)
        def wrap(*args, **kwargs):
            t0 = time.perf_counter()
            log.info("→ %s args=%s kwargs=%s", nm, _to_jsonable(args), _to_jsonable(kwargs))
            try:
                res = fn(*args, **kwargs)
                dt = (time.perf_counter() - t0) * 1000
                try:
                    preview = res if isinstance(res, (str, int, float, bool)) else type(res).__name__
                except Exception:
                    preview = "<res>"
                log.info("← %s ok in %.1f ms result=%s", nm, dt, preview)
                return res
            except Exception as e:
                dt = (time.perf_counter() - t0) * 1000
                tb = traceback.format_exc(limit=3)
                log.exception("← %s FAIL in %.1f ms: %s\n%s", nm, dt, e, tb)
                raise
        return wrap
    return deco


# pages/dem_diff_deck.py
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
    make_colorbar_datauri,
    plot_histogram,
    calculate_error_statistics,
    diff_to_base64_png,
    raster_bounds_ll,
)

MAIN_MAP_HEIGHT = 550
RIGHT_PANEL_WIDTH = 400

logging.basicConfig(level=os.getenv("PAGE_LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

TC_BASE = os.getenv("TERRACOTTA_PUBLIC_URL", "https://www.geohydroai.org/tc").rstrip("/")
MAPBOX_ACCESS_TOKEN = os.getenv("MAPBOX_ACCESS_TOKEN", "").strip()


dash.register_page(__name__, path="/dem-diff", name="DEM Diff (deck.gl)", order=2)

try:
    basin: gpd.GeoDataFrame = get_df("basin")
    basin = basin.to_crs("EPSG:4326")
    BASIN_JSON = json.loads(basin.to_json())
except Exception:
    BASIN_JSON = None

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

by_dem = defaultdict(list)
categories = set()
for l in layers_index:
    by_dem[l.get("dem")].append(l)
    if l.get("category"):
        categories.add(l["category"])

DEM_LIST = sorted([d for d in by_dem.keys() if d])
CATEGORY_LIST = sorted(categories)


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
        "@@type": "GeoJsonLayer",
        "id": "basin-outline",
        "data": geojson,
        "stroked": True,
        "filled": False,
        "getLineColor": [0, 102, 255, 200],
        "getLineWidth": 2,
        "lineWidthUnits": "pixels",
    }


def build_dem_url(colormap: str = "viridis") -> str:
    return f"{TC_BASE}/singleband/dem/fab_dem/{{z}}/{{x}}/{{y}}.png?colormap={colormap}&stretch_range=[250,2200]"


def build_spec(dem_url: str | None, diff_bitmap: dict | None, basin: dict | None,
               init_view=None, map_style: str = "mapbox://styles/mapbox/light-v11") -> str:
    layers = []
    if dem_url:
        layers.append(tile_layer("dem-tiles", dem_url, opacity=0.75))
    if diff_bitmap:
        layers.append(diff_bitmap)
    if basin:
        layers.append(basin_layer(basin))
    spec = {
        "mapStyle": map_style,
        "initialViewState": init_view or {"longitude": 25.03, "latitude": 47.8, "zoom": 8, "pitch": 0, "bearing": 0},
        "views": [
            {"@@type": "MapView", "id": "map"},
            {"@@type": "OrthographicView", "id": "hud", "flipY": True},
        ],
        "layers": layers,
    }
    return json.dumps(spec)


def hud_bitmap_layer(layer_id: str, image_data_uri: str, x: int, y: int, w: int, h: int) -> dict:
    return {
        "@@type": "BitmapLayer",
        "id": layer_id,
        "image": image_data_uri,
        "bounds": [x, y, x + w, y + h],
        "opacity": 1.0,
        "coordinateSystem": "cartesian",
        "viewId": "hud",
        "parameters": {"depthTest": False},
    }


def hud_text_layer(layer_id: str, lines: list[str], x: int, y: int, line_h: int = 16) -> dict:
    data = [{"text": t, "position": [x, y + i * line_h]} for i, t in enumerate(lines)]
    return {
        "@@type": "TextLayer",
        "id": layer_id,
        "data": data,
        "getText": "@@d.text",
        "getPosition": "@@d.position",
        "getSize": 12,
        "getColor": [230, 230, 230, 255],
        "fontFamily": "monospace",
        "background": True,
        "getBackgroundColor": [20, 20, 20, 180],
        "padding": 6,
        "coordinateSystem": "cartesian",
        "viewId": "hud",
        "parameters": {"depthTest": False},
    }


def hud_log_layer(lines: list[str], x: int = 12, y: int = 12, max_lines: int = 18) -> dict:
    return hud_text_layer("hud-log", lines[-max_lines:], x, y, 16)


def hud_state_layer(user_data: dict, x: int = 12, y: int = 320) -> dict:
    lines = ["STATE:"] + [f"{k}: {v}" for k, v in (user_data or {}).items()]
    return hud_text_layer("hud-state", lines, x, y, 16)


LEG_W, LEG_H = 22, 160
HIST_W, HIST_H = 360, 160
PAD = 12


def build_hud_layers(legend_uri: str, hist_uri: str, stats_lines: list[str], canvas_w: int, canvas_h: int):
    x_right = canvas_w - RIGHT_PANEL_WIDTH - PAD
    y_top = PAD
    layers = [
        hud_bitmap_layer("hud-legend", legend_uri, x_right, y_top, LEG_W, LEG_H),
        hud_bitmap_layer("hud-hist", hist_uri, x_right + LEG_W + PAD, y_top, HIST_W, HIST_H),
        hud_text_layer("hud-stats", stats_lines, x_right, y_top + LEG_H + PAD + HIST_H + PAD, 18),
    ]
    return layers


def _pick_path(name, category):
    arr = by_dem.get(name, [])
    if not arr:
        return None
    if category:
        for it in arr:
            if it.get("category") == category:
                return it.get("path")
    return arr[0].get("path")


def _rid():
    return uuid.uuid4().hex[:8]


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
        ], style={"width": "300px", "minWidth": "280px", "paddingRight": "16px"}),
        html.Div([
            dash_deckgl.DashDeckgl(
                id="deck-main",
                spec=build_spec(build_dem_url("viridis"), None, BASIN_JSON),
                height=MAIN_MAP_HEIGHT,
                cursor_position="bottom-right",
                events=["hover", "click", "keydown", "resize"],
                mapbox_key=MAPBOX_ACCESS_TOKEN,
            ),
            html.Div(id="deck-events", style={"fontFamily": "monospace", "marginTop": "6px"}),
        ], style={"width": "100%"}),
    ], style={"display": "grid", "gridTemplateColumns": "300px 1fr", "gap": "8px", "alignItems": "start"}),
])


@callback(
    Output("deck-main", "spec"),
    Output("state-store", "data"),
    Output("debug-log", "data"),
    Input("run", "n_clicks"),
    State("dem1", "value"),
    State("dem2", "value"),
    State("cat", "value"),
    State("debug-log", "data"),
    prevent_initial_call=True,
)
@trace("run_diff")
def run_diff(n, dem1, dem2, cat, dbg):
    rid = _rid()
    t0 = time.perf_counter()
    dbg = list(dbg or [])

    def push(msg):
        line = f"[{rid}] {msg}"
        logger.info(line)
        dbg.append(line)

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
        bounds = raster_bounds_ll(ref)
        diff_bitmap = bitmap_layer("diff-bitmap", img_uri, bounds)
        push("overlay ok")
    except Exception as e:
        push(f"overlay FAIL: {e}")
        return no_update, no_update, dbg

    legend_uri = make_colorbar_datauri(vmin, vmax, cmap="RdBu_r")
    hist_png = plot_histogram(diff, clip_range=(vmin, vmax))
    stats = calculate_error_statistics(diff)
    stats_lines = [
        f"{k}: {v:.3f}" if isinstance(v, (int, float)) and np.isfinite(v) else f"{k}: {v}"
        for k, v in stats.items()
    ]

    spec = build_spec(build_dem_url("terrain"), diff_bitmap, BASIN_JSON)
    spec_obj = json.loads(spec)

    hud_layers = []
    hud_layers += build_hud_layers(legend_uri, hist_png, stats_lines, canvas_w=1200, canvas_h=MAIN_MAP_HEIGHT)
    hud_layers.append(hud_log_layer(dbg))
    spec_obj["layers"].extend(hud_layers)

    spec_obj["userData"] = {
        "req_id": rid,
        "dem1": dem1,
        "dem2": dem2,
        "cat": cat,
        "vmin": vmin,
        "vmax": vmax,
        "cmap": "RdBu_r",
        "show_debug": True,
    }

    push(f"done {(time.perf_counter()-t0)*1000:.1f} ms")

    state = {"vmin": vmin, "vmax": vmax, "cmap": "RdBu_r", "dem1": dem1, "dem2": dem2, "cat": cat}

    return json.dumps(spec_obj), state, dbg


@callback(
    Output("deck-main", "spec"),
    Output("debug-log", "data"),
    Input("deck-main", "lastEvent"),
    State("deck-main", "spec"),
    State("debug-log", "data"),
    prevent_initial_call=True,
)
@trace("on_deck_event")
def on_deck_event(evt, spec_json, dbg):
    if not evt or not spec_json:
        return no_update, no_update
    dbg = list(dbg or [])

    typ = evt.get("eventType")
    info = evt.get("info") or {}
    layer_id = (info.get("layer") or {}).get("id")
    coord = info.get("coordinate")
    obj = info.get("object")
    key = (evt.get("key") or "").lower()

    line = f"[evt] {typ} layer={layer_id} coord={coord} picked={'yes' if obj else 'no'} key={key}"
    logger.info(line)
    dbg.append(line)

    spec = json.loads(spec_json)
    layers = spec.get("layers", [])

    replaced = False
    for i, lyr in enumerate(layers):
        if lyr.get("id") == "hud-log":
            layers[i] = hud_log_layer(dbg)
            replaced = True
            break
    if not replaced:
        layers.append(hud_log_layer(dbg))

    spec["layers"] = layers

    if typ == "keydown" and key == "d":
        show = not spec.get("userData", {}).get("show_debug", True)
        spec.setdefault("userData", {})["show_debug"] = show
        if not show:
            spec["layers"] = [lyr for lyr in spec["layers"] if lyr.get("id") != "hud-log"]
            dbg.append("[debug] HUD log OFF")
        else:
            spec["layers"].append(hud_log_layer(dbg))
            dbg.append("[debug] HUD log ON")

    return json.dumps(spec), dbg
