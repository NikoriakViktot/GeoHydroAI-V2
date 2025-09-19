# pages/dem_diff_deck.py
import os, json
import numpy as np
import geopandas as gpd
from collections import defaultdict

import dash
from dash import html, dcc, callback, Output, Input, State, no_update
import dash_deckgl

from utils.dem_tools import (
    compute_dem_difference, save_temp_diff_as_cog, make_colorbar_datauri,
    plot_histogram, calculate_error_statistics
)
from registry import get_df

# базовий URL до Terracotta (через nginx /tc)
TC_BASE = os.getenv("TERRACOTTA_PUBLIC_URL", "https://www.geohydroai.org/tc").rstrip("/")

dash.register_page(__name__, path="/dem-diff", name="DEM Diff (deck.gl)", order=2)

# ---- межа басейну
try:
    basin: gpd.GeoDataFrame = get_df("basin")
    print("Basin loaded! CRS:", basin.crs)
    basin = basin.to_crs("EPSG:4326")
    basin_json = json.loads(basin.to_json())
except Exception as e:
    print("❌ Error loading basin:", e)
    basin_json = None

# ---- layers_index.json
with open("assets/layers_index.json", "r") as f:
    layers_index = json.load(f)

by_dem = defaultdict(list)
categories = set()
for l in layers_index:
    by_dem[l["dem"]].append(l)
    categories.add(l["category"])
DEM_LIST = sorted(by_dem.keys())
CATEGORY_LIST = sorted(categories)

# ---- helpers
def build_dem_url(colormap="viridis"):
    # базовий DEM як фон (не diff)
    return f"{TC_BASE}/singleband/dem/fab_dem/{{z}}/{{x}}/{{y}}.png?colormap={colormap}&stretch_range=[250,2200]"

def build_tempdiff_url(basename, vmin, vmax, cmap="RdBu_r"):
    # у Terracotta /tiles/<basename>/... ми додаємо візуальні параметри (Terracotta це підтримує у твоїй конфі)
    return f"{TC_BASE}/tiles/{basename}/{{z}}/{{x}}/{{y}}.png?colormap={cmap}&stretch_range=[{vmin:.3f},{vmax:.3f}]"

def tile_layer(layer_id: str, url: str, opacity: float = 1.0) -> dict:
    return {
        "@@type": "TileLayer",
        "id": layer_id,
        "data": url,
        "minZoom": 0, "maxZoom": 19, "tileSize": 256, "opacity": opacity,
        "renderSubLayers": {
            "@@function": ["tile", {
                "type":"BitmapLayer",
                "id": f"{layer_id}-bitmap",
                "image":"@@tile.data", "bounds":"@@tile.bbox",
                "opacity": opacity
            }]
        },
    }

def basin_layer(geojson: dict) -> dict:
    return {
        "@@type":"GeoJsonLayer","id":"basin-outline",
        "data": geojson, "stroked": True, "filled": False,
        "getLineColor":[0,102,255,200], "getLineWidth":2, "lineWidthUnits":"pixels"
    }

def build_spec(dem_url: str | None, diff_url: str | None, basin: dict | None, init_view=None) -> str:
    layers = []
    if dem_url: layers.append(tile_layer("dem-tiles", dem_url, opacity=0.75))
    if diff_url: layers.append(tile_layer("diff-tiles", diff_url, opacity=0.95))
    if basin: layers.append(basin_layer(basin))
    spec = {
        "initialViewState": init_view or {"longitude":25.03,"latitude":47.8,"zoom":10,"pitch":0,"bearing":0},
        "layers": layers
    }
    return json.dumps(spec)

# ---- UI
layout = html.Div([
    html.H3("DEM Difference Analysis (deck.gl + Terracotta)"),
    html.Div([
        html.Div([
            html.Label("DEM 1"), dcc.Dropdown(id="dem1", options=[{"label":d,"value":d} for d in DEM_LIST]),
            html.Label("DEM 2"), dcc.Dropdown(id="dem2", options=[{"label":d,"value":d} for d in DEM_LIST]),
            html.Label("Категорія"), dcc.Dropdown(id="cat", options=[{"label":c,"value":c} for c in CATEGORY_LIST]),
            html.Br(),
            html.Button("Порахувати різницю", id="run"),
        ], style={"width":"340px","marginRight":"16px","display":"inline-block","verticalAlign":"top"}),
        html.Div([
            dash_deckgl.DashDeckgl(
                id="deck-main",
                spec=build_spec(build_dem_url("viridis"), None, basin_json),
                description={"top-right": "<div id='legend'>Legend</div>"},
                height=640, cursor_position="bottom-right", events=["hover"]
            ),
            html.Div(id="deck-events", style={"fontFamily":"monospace","marginTop":"6px"})
        ], style={"width":"calc(100% - 360px)","display":"inline-block","verticalAlign":"top"})
    ]),
    html.Div([
        html.Div([
            html.H4("Гістограма"), html.Img(id="hist", style={"height":"220px"})
        ], style={"display":"inline-block","marginRight":"24px"}),
        html.Div(id="stats", style={"display":"inline-block","verticalAlign":"top","fontFamily":"monospace"})
    ], style={"marginTop":"14px"}),

    html.Hr(),
    html.H4("Режим порівняння (дві панелі)"),
    html.Div([
        dash_deckgl.DashDeckgl(
            id="deck-left",
            spec=build_spec(build_dem_url("terrain"), None, basin_json),
            height=420, cursor_position="none"
        ),
        dash_deckgl.DashDeckgl(
            id="deck-right",
            spec=build_spec(build_dem_url("viridis"), None, basin_json),
            height=420, cursor_position="none"
        ),
    ], style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"8px"})
])

# ---- колбек: основний аналіз
@callback(
    Output("deck-main","spec"),
    Output("hist","src"),
    Output("stats","children"),
    Input("run","n_clicks"),
    State("dem1","value"), State("dem2","value"), State("cat","value"),
    prevent_initial_call=True
)
def run_diff(n, dem1, dem2, cat):
    if not dem1 or not dem2 or dem1 == dem2:
        return no_update, no_update, "Оберіть різні DEM!"

    def pick_path(name, category):
        arr = by_dem.get(name, [])
        if category:
            for it in arr:
                if it.get("category") == category:
                    return it.get("path")
        return arr[0]["path"] if arr else None

    p1, p2 = pick_path(dem1, cat), pick_path(dem2, cat)
    if not p1 or not p2:
        return no_update, no_update, "DEM не знайдено у layers_index!"

    try:
        diff, ref = compute_dem_difference(p1, p2)
    except Exception as e:
        return no_update, no_update, f"Помилка при обчисленні: {e}"

    try:
        q1, q99 = np.nanpercentile(diff, [1, 99])
        vmin, vmax = float(q1), float(q99)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = -10.0, 10.0
    except Exception:
        vmin, vmax = -10.0, 10.0

    try:
        cog_path = save_temp_diff_as_cog(diff, ref, prefix="demdiff_")
        basename = os.path.basename(cog_path)
        diff_url = build_tempdiff_url(basename, vmin, vmax, cmap="RdBu_r")
    except Exception as e:
        return no_update, no_update, f"Помилка збереження COG/URL: {e}"

    legend_uri = make_colorbar_datauri(vmin, vmax, cmap="RdBu_r")
    legend_html = f"<img src='{legend_uri}' style='height:160px'/>"

    spec = build_spec(build_dem_url("terrain"), diff_url, basin_json)
    hist = plot_histogram(diff, clip_range=(vmin, vmax))
    stats = calculate_error_statistics(diff)

    rows = [
        html.Tr([html.Th(k), html.Td(f"{v:.3f}" if isinstance(v, float) and np.isfinite(v) else v)])
        for k, v in stats.items()
    ]
    stats_tbl = html.Table(
        rows,
        style={"background": "#181818", "color": "#eee", "padding": "6px"}
    )

    spec_obj = json.loads(spec)
    spec_obj.setdefault("description", {})["top-right"] = legend_html
    return json.dumps(spec_obj), hist, stats_tbl


# (опційно) події
@callback(Output("deck-events","children"), Input("deck-main","lastEvent"))
def show_evt(evt):
    if not evt: return ""
    return f"{evt.get('eventType')} @ {evt.get('coordinate')}"
