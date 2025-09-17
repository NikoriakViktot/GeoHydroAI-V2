# layout/metadata_groups.py
"""
Этот модуль отвечает за загрузку и группировку метаданных,
а также предоставляет словари лейблов и тултипов.
"""
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

# -------------------------------------------------------------------------------------------------
#  DEBUG
# -------------------------------------------------------------------------------------------------
DEBUG = os.environ.get("GH_DEBUG", "1").lower() not in ("0", "false", "no")

def debug(*args: Any) -> None:
    if DEBUG:
        print("[metadata_groups]", *args, flush=True)

# -------------------------------------------------------------------------------------------------
#  METADATA LOADING
# -------------------------------------------------------------------------------------------------
_metadata_cache: Optional[List[Dict[str, Any]]] = None

def get_metadata(path: str = "assets/metadata.json") -> List[Dict[str, Any]]:
    """
    Lazy-load metadata.json and cache it.
    """
    global _metadata_cache
    if _metadata_cache is None:
        debug(f"Loading metadata from {path}...")
        with open(path, encoding="utf-8") as f:
            _metadata_cache = json.load(f)
        debug(f"Loaded {_metadata_cache and len(_metadata_cache)} records.")
    return _metadata_cache  # type: ignore

# -------------------------------------------------------------------------------------------------
#  CATEGORY <-> GROUP MAPPINGS
# -------------------------------------------------------------------------------------------------
CATEGORY_TO_GROUP: Dict[str, str] = {
    # Relief basic
    "aspect": "relief_basic",
    "slope_horn": "relief_basic",
    "slope_whitebox": "relief_basic",
    "curvature": "relief_basic",
    # Morphometric
    "tpi": "morphometric",
    "tri": "morphometric",
    "roughness": "morphometric",
    "geomorphons": "morphometric",
    # Hydrological indices
    "twi": "hydro_indices",
    "distance_to_stream": "hydro_indices",
    "hand": "hydro_indices",
    "hand_500": "hydro_indices",
    "hand_1000": "hydro_indices",
    "hand_1500": "hydro_indices",
    "hand_2000": "hydro_indices",
    # Flood scenarios
    "flood_scenarios": "flood",
    # Flow
    "d8_accum": "hydro_flow",
    "d8_pointer": "hydro_flow",
    "stream_raster": "hydro_flow",
    "stream_raster_500": "hydro_flow",
    "stream_raster_1000": "hydro_flow",
    "stream_raster_1500": "hydro_flow",
    "stream_raster_2000": "hydro_flow",
    # DEM prep
    "breached": "dem_prep",
}

GROUP_LABEL: Dict[str, str] = {
    "relief_basic": "Рельєф: базові похідні",
    "morphometric": "Морфометричні індекси",
    "hydro_indices": "Гідрологічні індекси",
    "flood": "Паводки / Затоплення",
    "hydro_flow": "Гідромережа та стік",
    "dem_prep": "Гідрокоригований DEM",
}

CAT_LABEL: Dict[str, str] = {
    "aspect": "Aspect (експозиція)",
    "slope_horn": "Slope (Horn)",
    "slope_whitebox": "Slope (WB)",
    "curvature": "Curvature (кривина)",
    "tpi": "TPI (позиція)",
    "tri": "TRI (шорсткість)",
    "roughness": "Roughness",  # etc.
    "geomorphons": "Geomorphons",
    "twi": "TWI (вологість)",
    "distance_to_stream": "Dist. to stream",
    "hand": "HAND",
    "hand_500": "HAND 500m",
    "hand_1000": "HAND 1000m",
    "hand_1500": "HAND 1500m",
    "hand_2000": "HAND 2000m",
    "flood_scenarios": "Flood scenarios",
    "d8_accum": "Flow accum",
    "d8_pointer": "Flow dir",
    "stream_raster": "Streams",
    "stream_raster_500": "Streams ≥500",
    "stream_raster_1000": "Streams ≥1000",
    "stream_raster_1500": "Streams ≥1500",
    "stream_raster_2000": "Streams ≥2000",
    "breached": "Hydro‑conditioned DEM",
}

# -------------------------------------------------------------------------------------------------
#  UTILS
# -------------------------------------------------------------------------------------------------
def parse_flood_depth(record: Dict[str, Any]) -> Optional[int]:
    """
    Extract integer depth (meters) from record['flood'], e.g. "5m" -> 5.
    """
    val = record.get("flood")
    if not isinstance(val, str):
        return None
    try:
        return int(val.strip().lower().rstrip("m"))
    except ValueError:
        debug(f"Failed to parse flood depth from {val}")
        return None

# -------------------------------------------------------------------------------------------------
#  GROUPING FUNCTIONS
# -------------------------------------------------------------------------------------------------
def group_metadata_by_dem(dem_name: str, metadata: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Return mapping {group_id: [records]} for a given DEM.
    Excludes 'dem' and 'lulc' categories.
    """
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in metadata:
        cat = rec.get("category")
        if cat in ("dem", "lulc") or rec.get("dem") != dem_name:
            continue
        gid = CATEGORY_TO_GROUP.get(cat)
        if gid:
            grouped[gid].append(rec)
    return grouped


# ============================================================================
# layout/sidebar_auto.py
# ============================================================================
"""
Этот модуль строит статическую часть сайдбара и
инициализирует Stores для динамических контролей.
"""
import os
from typing import List, Dict, Any

from dash import html, dcc
from dash.dependencies import Input, Output, State, ALL
import dash_leaflet as dl



# -------------------------------------------------------------------------------------------------
#  DEBUG
# -------------------------------------------------------------------------------------------------
DEBUG = os.environ.get("GH_DEBUG", "1").lower() not in ("0", "false", "no")

def debug_sb(*args: Any) -> None:
    if DEBUG:
        print("[sidebar_auto]", *args, flush=True)

# -------------------------------------------------------------------------------------------------
#  STYLES
# -------------------------------------------------------------------------------------------------
SIDEBAR_WIDTH = 300
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "width": f"{SIDEBAR_WIDTH}px",
    "height": "100vh",
    "padding": "1rem",
    "backgroundColor": "#2E3B4E",
    "color": "#EEE",
    "overflowY": "auto",
}
LABEL_STYLE = {"marginTop": "1rem", "color": "#FFF"}
DROPDOWN_STYLE = {"width": "100%"}

# -------------------------------------------------------------------------------------------------
#  SIDEBAR BUILDER
# -------------------------------------------------------------------------------------------------
def make_sidebar_auto() -> html.Div:
    metadata = get_metadata()

    # DEM select ---------------------------------------------------------------
    dem_records = [m for m in metadata if m.get("category") == "dem"]
    dem_options = [
        {"label": rec["name"].replace("_", " ").upper(), "value": rec["name"]}
        for rec in dem_records
    ]
    default_dem = dem_options[0]["value"] if dem_options else None

    # group select -------------------------------------------------------------
    group_options = [
        {"label": GROUP_LABEL[gid], "value": gid}
        for gid in GROUP_LABEL
    ]

    # LULC select --------------------------------------------------------------
    lulc_opts = [
        {"label": r["name"].split("_",1)[-1].upper(), "value": r["name"]}
        for r in metadata if r.get("category") == "lulc"
    ]

    return html.Div([
        html.H4("🔧 Фільтри", style={"color":"#FFF"}),
        html.Label("Base DEM:", style=LABEL_STYLE),
        dcc.Dropdown(
            id="dem-select", options=dem_options, value=default_dem,
            clearable=False, style=DROPDOWN_STYLE
        ),

        html.Label("Derived group:", style=LABEL_STYLE),
        dcc.Dropdown(
            id="derived-group", options=group_options,
            placeholder="Select group...", style=DROPDOWN_STYLE
        ),
        html.Div(id="derived-group-controls"),

        html.Label("LULC:", style=LABEL_STYLE),
        dcc.Dropdown(
            id="lulc-select", options=lulc_opts,
            multi=True, style=DROPDOWN_STYLE
        ),

        # Stores for callbacks
        dcc.Store(id="metadata-store", data=metadata),
        dcc.Store(id="selected-derived-store"),
        dcc.Store(id="selected-flood-store"),

        # Optional debug
        html.Pre(id="sidebar-debug", style={"color":"#0f0","background":"#111"}) if DEBUG else html.Div()
    ], style=SIDEBAR_STYLE)

