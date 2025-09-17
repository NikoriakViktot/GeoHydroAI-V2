# pages/dem_map_auto.py
import os
import json
from typing import Dict, Any, Optional, List

import geopandas as gpd
import dash
import dash_leaflet as dl
from dash import html, dcc, callback, Input, Output, State

from layout.sidebar_metadata_auto import (
    get_metadata,
    parse_flood_depth,
    build_options_for_group,
    make_sidebar_auto,
    GROUP_LABEL,
)

print("[dem_map_auto] importing")

# ---------------------------------------------------------------------------
# Page registration
# ---------------------------------------------------------------------------
dash.register_page(
    __name__,
    path="/dem-map-auto",
    description="GeoHydroAI | Map DEM",
    name="Maps DEM",
    title="GeoHydroAI | Map DEM",
    order=2,
)

# ---------------------------------------------------------------------------
# Debug flag
# ---------------------------------------------------------------------------
DEBUG = os.environ.get("GH_DEBUG", "1").lower() not in ("0", "false", "no")
def dbg(*args, **kwargs):
    if DEBUG:
        print("[dem_map_auto]", *args, **kwargs, flush=True)

# ---------------------------------------------------------------------------
# Basin (optional overlay)
# ---------------------------------------------------------------------------
def load_basin_geojson():
    try:
        gdf = gpd.read_file("data/basin_bil_cher_4326.gpkg").to_crs("EPSG:4326")
        print("[dem_map_auto] basin loaded; features:", len(gdf))
        return json.loads(gdf.to_json())
    except Exception as e:
        print("[dem_map_auto] basin load FAILED:", e)
        return {"type": "FeatureCollection", "features": []}

basin_geojson = load_basin_geojson()

# ---------------------------------------------------------------------------
# Colormaps
# ---------------------------------------------------------------------------
COLORMAPS = ["viridis", "terrain", "inferno", "jet", "spectral", "rainbow"]
DEFAULT_DEM_STRETCH = "[0,2200]"   # TODO: автомат. статистика

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
sidebar = make_sidebar_auto()

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
layout = html.Div(
    [
        html.Div(sidebar, style={"flex": "0 0 300px"}),
        html.Div(
            [
                dcc.Dropdown(
                    id="colormap-dropdown",
                    options=[{"label": c, "value": c} for c in COLORMAPS],
                    value="viridis",
                    style={"margin-bottom": "1rem"},
                ),
                dl.Map(
                    id="main-map",
                    center=[47.8, 25.03],
                    zoom=10,
                    style={"width": "100%", "height": "700px"},
                    children=[
                        dl.LayersControl(
                            [
                                dl.BaseLayer(
                                    dl.TileLayer(
                                        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
                                        attribution="© OpenStreetMap contributors",
                                    ),
                                    name="OSM",
                                    checked=True,
                                ),
                                dl.Overlay(
                                    dl.TileLayer(id="dem-tile", opacity=0.8, zIndex=400),
                                    id="dem-overlay",
                                    name="DEM",
                                    checked=True,
                                ),
                                dl.Overlay(
                                    dl.LayerGroup(id="derived-layers"),
                                    id="derived-overlay",
                                    name="Derived",
                                    checked=False,
                                ),
                                dl.Overlay(
                                    dl.LayerGroup(id="flood-layers"),
                                    id="flood-overlay",
                                    name="Flood",
                                    checked=False,
                                ),
                                dl.Overlay(
                                    dl.LayerGroup(id="lulc-layers"),
                                    id="lulc-overlay",
                                    name="LULC",
                                    checked=False,
                                ),
                                dl.Overlay(
                                    dl.GeoJSON(
                                        id="basin-geojson",
                                        data=basin_geojson,
                                        options={"style": {"color": "cyan", "weight": 2, "fill": False}},
                                    ),
                                    id="basin-overlay",
                                    name="Basin",
                                    checked=True,
                                ),
                            ],
                            position="topleft",
                            collapsed=False,
                        )
                    ],
                ),
                dcc.Store(id="dem-map-debug-store"),
                html.Pre(
                    id="dem-map-debug-pre",
                    style={
                        "marginTop": "1rem",
                        "whiteSpace": "pre-wrap",
                        "fontSize": "10px",
                        "maxHeight": "160px",
                        "overflowY": "auto",
                        "border": "1px solid #444",
                        "padding": "4px",
                        "backgroundColor": "#111",
                        "color": "#0f0",
                    },
                ) if DEBUG else html.Div(),
            ],
            style={"flex": "1 1 auto", "padding": "0 1rem"},
        ),
    ],
    style={"display": "flex", "height": "100vh"},
)

# ---------------------------------------------------------------------------
# URL BUILDER
# ---------------------------------------------------------------------------
def build_tile_url(
    record: Dict[str, Any],
    cmap: Optional[str] = None,
    stretch: Optional[str] = None,
) -> Optional[str]:
    """
    Побудувати tile URL для /tc/singleband/<category>/<name>/{z}/{x}/{y}.png
    """
    if not record:
        return None
    cat = record.get("category")
    nm = record.get("name")
    if not cat or not nm:
        return None

    url = f"/tc/singleband/{cat}/{nm}/{{z}}/{{x}}/{{y}}.png"
    params = []

    if cat == "dem":
        params.append(f"colormap={cmap or 'viridis'}")
        params.append(f"stretch_range={stretch or DEFAULT_DEM_STRETCH}")
    elif cat in (
        "aspect", "slope_horn", "slope_whitebox", "curvature",
        "tpi", "tri", "roughness", "twi", "distance_to_stream",
        "hand", "hand_500", "hand_1000", "hand_1500", "hand_2000",
        "d8_accum",
    ):
        if cmap:
            params.append(f"colormap={cmap}")
        if stretch:
            params.append(f"stretch_range={stretch}")
    # інші (flood_scenarios, lulc, geomorphons, d8_pointer, stream_raster*, breached) без параметрів

    if params:
        url += "?" + "&".join(params)
    return url


# =============================================================================
# SIDEBAR BEHAVIOR
# =============================================================================

@callback(
    Output("derived-group-controls", "children"),
    Output("derived-layer-multi", "style"),
    Output("flood-visible", "style"),
    Output("flood-depth-range", "style"),
    Output("derived-layer-multi", "options"),
    Input("derived-group", "value"),
    Input("dem-select", "value"),
    State("metadata-store", "data"),
)
def render_group_controls(gid, dem_name, metadata):
    """
    Керуємо тим, що бачить користувач під групою:
    - Якщо gid == flood -> показуємо flood-visible + flood-range.
    - Інакше -> показуємо derived-layer-multi + підставляємо опції для (gid, dem).
    """
    metadata = metadata or get_metadata()
    if not gid:
        # нічого не вибрано
        return (html.Div("Оберіть групу.", style={"color": "#CCC"}),
                {"display": "none"}, {"display": "none"}, {"display": "none"}, [])

    if gid == "flood":
        # flood UI
        children = html.Div(
            [
                html.Span("Паводки (глибина, м):", style={"color": "#DDD"}),
                # фактичні елементи вже є в layout; тут лише декоративний текст
            ],
            style={"marginBottom": "0.25rem"},
        )
        return children, {"display": "none"}, {"display": "block", "color": "#DDD"}, {"display": "block"}, []

    # звичайні похідні
    opts = build_options_for_group(dem_name, gid, metadata)
    children = html.Div(
        [
            html.Span(GROUP_LABEL.get(gid, gid), style={"color": "#DDD"}),
        ],
        style={"marginBottom": "0.25rem"},
    )
    return children, {"display": "block"}, {"display": "none"}, {"display": "none"}, opts


# -- Збирання вибраних похідних ------------------------------------------------
@callback(
    Output("selected-derived-store", "data"),
    Input("derived-layer-multi", "value"),
)
def collect_selected_derived(values):
    if not values:
        return []
    return values


# -- Збирання паводків --------------------------------------------------------
@callback(
    Output("selected-flood-store", "data"),
    Input("flood-visible", "value"),
    Input("flood-depth-range", "value"),
    State("dem-select", "value"),
    State("metadata-store", "data"),
)
def collect_selected_flood(show_vals, depth_range, dem_name, metadata):
    if not show_vals:
        return []
    metadata = metadata or get_metadata()
    lo, hi = depth_range
    recs = [
        m for m in metadata
        if m.get("dem") == dem_name and m.get("category") == "flood_scenarios"
    ]
    out = []
    for r in recs:
        d = parse_flood_depth(r)
        if d is not None and lo <= d <= hi:
            out.append(r["name"])
    return sorted(out)


# =============================================================================
# MAP UPDATES
# =============================================================================

# -- DEM tile -----------------------------------------------------------------
@callback(
    Output("dem-tile", "url"),
    Output("dem-overlay", "checked"),
    Output("dem-map-debug-store", "data"),
    Input("colormap-dropdown", "value"),
    Input("dem-select", "value"),
    State("metadata-store", "data"),
    State("derived-overlay", "checked"),
    State("flood-overlay", "checked"),
    State("lulc-overlay", "checked"),
    prevent_initial_call=False,
)
def upd_dem_tile(cmap, dem_name, metadata, derived_on, flood_on, lulc_on):
    metadata = metadata or get_metadata()
    by_name = {m["name"]: m for m in metadata}
    rec = by_name.get(dem_name) or next((m for m in metadata if m["category"] == "dem"), None)

    url = build_tile_url(rec, cmap=cmap, stretch=DEFAULT_DEM_STRETCH)

    # авто-ховати DEM якщо будь-який інший overlay активний
    dem_checked = not (derived_on or flood_on or lulc_on)

    dbg_info = {
        "trig": "dem/cmap change",
        "cmap": cmap,
        "dem": dem_name,
        "url": url,
        "derived_on": derived_on,
        "flood_on": flood_on,
        "lulc_on": lulc_on,
        "dem_checked": dem_checked,
    }
    dbg("DEM URL ->", url)
    return url, dem_checked, dbg_info


# -- Derived layers -----------------------------------------------------------
@callback(
    Output("derived-layers", "children"),
    Output("derived-overlay", "checked"),
    Input("selected-derived-store", "data"),
    State("colormap-dropdown", "value"),
    State("metadata-store", "data"),
)
def upd_derived_layers(selected_names, cmap, metadata):
    metadata = metadata or get_metadata()
    if not selected_names:
        return [], False
    by_name = {m["name"]: m for m in metadata}
    tiles = []
    for nm in selected_names:
        rec = by_name.get(nm)
        if rec is None:
            continue
        url = build_tile_url(rec, cmap=None, stretch=None)  # можна cmap
        tiles.append(dl.TileLayer(url=url, opacity=0.7, zIndex=420))
        dbg("DERIVED add:", nm, "->", url)
    return tiles, True


# -- Flood layers -------------------------------------------------------------
@callback(
    Output("flood-layers", "children"),
    Output("flood-overlay", "checked"),
    Input("selected-flood-store", "data"),
    State("metadata-store", "data"),
)
def upd_flood_layers(flood_names, metadata):
    metadata = metadata or get_metadata()
    if not flood_names:
        return [], False
    by_name = {m["name"]: m for m in metadata}
    tiles = []
    for nm in flood_names:
        rec = by_name.get(nm)
        if rec is None:
            continue
        url = build_tile_url(rec, cmap=None, stretch=None)
        tiles.append(dl.TileLayer(url=url, opacity=0.5, zIndex=450))
        dbg("FLOOD add:", nm, "->", url)
    return tiles, True


# -- LULC layers --------------------------------------------------------------
@callback(
    Output("lulc-layers", "children"),
    Output("lulc-overlay", "checked"),
    Input("lulc-select", "value"),
    State("metadata-store", "data"),
)
def upd_lulc_layers(lulc_names, metadata):
    metadata = metadata or get_metadata()
    if not lulc_names:
        return [], False
    by_name = {m["name"]: m for m in metadata}
    tiles = []
    for nm in lulc_names:
        rec = by_name.get(nm)
        if rec is None:
            continue
        url = build_tile_url(rec, cmap=None, stretch=None)
        tiles.append(dl.TileLayer(url=url, opacity=0.7, zIndex=500))
        dbg("LULC add:", nm, "->", url)
    return tiles, True


# -- Debug text ---------------------------------------------------------------
@callback(
    Output("dem-map-debug-pre", "children"),
    Input("dem-map-debug-store", "data"),
    Input("derived-layers", "children"),
    Input("flood-layers", "children"),
    Input("lulc-layers", "children"),
    prevent_initial_call=False,
)
def show_debug(dem_dbg, derived_children, flood_children, lulc_children):
    if not DEBUG:
        return ""
    import pprint
    d = {
        "dem_dbg": dem_dbg,
        "derived_n": len(derived_children or []),
        "flood_n": len(flood_children or []),
        "lulc_n": len(lulc_children or []),
    }
    return pprint.pformat(d, width=70)
