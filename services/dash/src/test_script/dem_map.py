# ============================================================================
# pages/dem_map_auto.py  (виправлено)
# ============================================================================

import os, json
from typing import Dict, Any, Optional, List
import geopandas as gpd
import dash
import dash_leaflet as dl
from dash import html, dcc, callback, Input, Output, State, ALL

from layout.sidebar_metadata import (
    get_metadata,
    make_sidebar_auto,
    parse_flood_depth,
)

print("[dem_map_auto] importing")

dash.register_page(
    __name__,
    path="/dem-map",
    name="Maps DEM",
    title="GeoHydroAI | Map DEM",
    order=2,
)

DEBUG = os.environ.get("GH_DEBUG", "1").lower() not in ("0","false","no")
def dbg(*args, **kwargs):
    if DEBUG:
        print("[dem_map_auto]", *args, **kwargs, flush=True)

# ----------------------------------------------------------------------------
# basin outline
# ----------------------------------------------------------------------------
def load_basin_geojson():
    try:
        gdf = gpd.read_file("data/basin_bil_cher_4326.gpkg").to_crs("EPSG:4326")
        print("[dem_map_auto] basin loaded; features:", len(gdf))
        return json.loads(gdf.to_json())
    except Exception as e:
        print("[dem_map_auto] basin load FAILED:", e)
        return {"type":"FeatureCollection","features":[]}

basin_geojson = load_basin_geojson()

# ----------------------------------------------------------------------------
# colormaps
# ----------------------------------------------------------------------------
COLORMAPS = ["viridis","terrain","inferno","jet","spectral","rainbow"]

# ----------------------------------------------------------------------------
# sidebar
# ----------------------------------------------------------------------------
sidebar = make_sidebar_auto()

# ----------------------------------------------------------------------------
# layout
# ----------------------------------------------------------------------------
layout = html.Div(
    [
        html.Div(sidebar, style={"flex":"0 0 300px"}),
        html.Div(
            [
                dcc.Dropdown(
                    id="colormap-dropdown",
                    options=[{"label":c,"value":c} for c in COLORMAPS],
                    value="viridis",
                    style={"margin-bottom":"1rem"},
                ),
                dl.Map(
                    id="main-map",
                    center=[47.8,25.03],
                    zoom=10,
                    style={"width":"100%","height":"700px"},
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
                                dl.Overlay(  # basin outline (опція)
                                    dl.GeoJSON(
                                        id="basin-geojson",
                                        data=basin_geojson,
                                        options={"style":{"color":"cyan","weight":2,"fill":False}},
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
                        "marginTop":"1rem",
                        "whiteSpace":"pre-wrap",
                        "fontSize":"10px",
                        "maxHeight":"160px",
                        "overflowY":"auto",
                        "border":"1px solid #444",
                        "padding":"4px",
                        "backgroundColor":"#111",
                        "color":"#0f0",
                    },
                ) if DEBUG else html.Div(),
            ],
            style={"flex":"1 1 auto","padding":"0 1rem"},
        ),
    ],
    style={"display":"flex","height":"100vh"},
)

# ----------------------------------------------------------------------------
# URL builder
# ----------------------------------------------------------------------------
DEFAULT_DEM_STRETCH = "[0,2200]"

def build_tile_url(
    record: Dict[str, Any],
    cmap: Optional[str] = None,
    stretch: Optional[str] = None,
) -> Optional[str]:
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
        "aspect","slope_horn","slope_whitebox","curvature",
        "tpi","tri","roughness","twi","distance_to_stream",
        "hand","hand_500","hand_1000","hand_1500","hand_2000",
        "d8_accum",
    ):
        if cmap: params.append(f"colormap={cmap}")
        if stretch: params.append(f"stretch_range={stretch}")
    # інші (flood_scenarios, lulc, geomorphons, d8_pointer, stream_raster*, breached) без параметрів
    if params:
        url += "?" + "&".join(params)
    return url

# ============================================================================
#from dash import callback, Input, Output, State, ALL
import dash_leaflet as dl
from layout.sidebar_metadata import get_metadata, parse_flood_depth

# -----------------------------------------------------------------------------
# 1) Динамічна генерація контролів для обраної групи
# -----------------------------------------------------------------------------
@callback(
    Output("derived-group-controls", "children"),
    Input("derived-group", "value"),
    Input("dem-select", "value"),
    State("metadata-store", "data"),
)
def render_group_controls(gid, dem_name, metadata):
    metadata = metadata or get_metadata()
    if not gid:
        return html.Div("Оберіть групу.", style={"color":"#CCC"})
    return build_group_controls(gid, dem_name, metadata)


# -----------------------------------------------------------------------------
# 2) Збір вибраних похідних шарів (усі групи, крім flood)
# -----------------------------------------------------------------------------
@callback(
    Output("selected-derived-store", "data"),
    Input({"type": "derived-layer-select", "group": ALL}, "value"),
)
def collect_selected_derived(values_list):
    sel = []
    for v in values_list:
        if v:
            sel.extend(v)
    return sorted(set(sel))


# -----------------------------------------------------------------------------
# 3) Збір вибраних flood‑сценаріїв
# -----------------------------------------------------------------------------
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
    lo, hi = depth_range or (None, None)
    recs = [
        m for m in metadata
        if m["dem"] == dem_name and m["category"] == "flood_scenarios"
    ]
    out = []
    for r in recs:
        d = parse_flood_depth(r)
        if d is not None and lo <= d <= hi:
            out.append(r["name"])
    return sorted(out)


# -----------------------------------------------------------------------------
# 4) Оновлення DEM‑шару (URL + видимість overlay)
# -----------------------------------------------------------------------------
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
)
def update_dem(cmap, dem_name, metadata, derived_on, flood_on, lulc_on):
    metadata = metadata or get_metadata()
    by_name = {m["name"]: m for m in metadata}
    rec = by_name.get(dem_name) or next(m for m in metadata if m["category"] == "dem")
    url = build_tile_url(rec, cmap=cmap, stretch=DEFAULT_DEM_STRETCH)
    # Авто‑сховати DEM, якщо є хоча б один overlay
    dem_checked = not (derived_on or flood_on or lulc_on)
    debug = {
        "url": url,
        "derived_on": derived_on,
        "flood_on": flood_on,
        "lulc_on": lulc_on,
        "dem_checked": dem_checked
    }
    return url, dem_checked, debug


# -----------------------------------------------------------------------------
# 5) Оновлення Derived‑шарів (не flood)
# -----------------------------------------------------------------------------
@callback(
    Output("derived-layers", "children"),
    Output("derived-overlay", "checked"),
    Input("selected-derived-store", "data"),
    State("colormap-dropdown", "value"),
    State("metadata-store", "data"),
)
def update_derived_layers(selected, cmap, metadata):
    metadata = metadata or get_metadata()
    if not selected:
        return [], False
    by_name = {m["name"]: m for m in metadata}
    tiles = []
    for name in selected:
        rec = by_name.get(name)
        if not rec:
            continue
        url = build_tile_url(rec, cmap=None, stretch=None)
        tiles.append(dl.TileLayer(url=url, opacity=0.7, zIndex=420))
    return tiles, True


# -----------------------------------------------------------------------------
# 6) Оновлення Flood‑шарів (в окремий Overlay)
# -----------------------------------------------------------------------------
@callback(
    Output("flood-layers", "children"),
    Output("flood-overlay", "checked"),
    Input("selected-flood-store", "data"),
    State("metadata-store", "data"),
)
def update_flood_layers(selected, metadata):
    metadata = metadata or get_metadata()
    if not selected:
        return [], False
    by_name = {m["name"]: m for m in metadata}
    tiles = []
    for name in selected:
        rec = by_name.get(name)
        if not rec:
            continue
        url = build_tile_url(rec, cmap=None, stretch=None)
        tiles.append(dl.TileLayer(url=url, opacity=0.5, zIndex=450))
    return tiles, True


# -----------------------------------------------------------------------------
# 7) Оновлення LULC‑шарів
# -----------------------------------------------------------------------------
@callback(
    Output("lulc-layers", "children"),
    Output("lulc-overlay", "checked"),
    Input("lulc-select", "value"),
    State("metadata-store", "data"),
)
def update_lulc_layers(selected, metadata):
    metadata = metadata or get_metadata()
    if not selected:
        return [], False
    by_name = {m["name"]: m for m in metadata}
    tiles = []
    for name in selected:
        rec = by_name.get(name)
        if not rec:
            continue
        url = build_tile_url(rec, cmap=None, stretch=None)
        tiles.append(dl.TileLayer(url=url, opacity=0.7, zIndex=500))
    return tiles, True


# -----------------------------------------------------------------------------
# 8) Відображення debug‑інформації
# -----------------------------------------------------------------------------
@callback(
    Output("dem-map-debug-pre", "children"),
    Input("dem-map-debug-store", "data"),
    Input("derived-layers", "children"),
    Input("flood-layers", "children"),
    Input("lulc-layers", "children"),
)
def show_debug(dbg, derived_children, flood_children, lulc_children):
    if not DEBUG:
        return ""
    import pprint
    d = {
        "dem_info": dbg,
        "derived_count": len(derived_children or []),
        "flood_count": len(flood_children or []),
        "lulc_count": len(lulc_children or []),
    }
    return pprint.pformat(d, width=60)
