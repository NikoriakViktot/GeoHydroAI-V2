# layout/sidebar_metadata_auto.py
import json
import os
import re
from functools import lru_cache
from typing import Dict, List, Any, Optional, Tuple

from dash import html, dcc

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
METADATA_PATHS = [
    "assets/metadata.json",  # основне місце
    "metadata.json",         # fallback поруч
]

# ---------------------------------------------------------------------------
# LOAD METADATA
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_metadata() -> List[Dict[str, Any]]:
    """Load and cache metadata records."""
    for p in METADATA_PATHS:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            # нормалізуємо: переконуємось, що є dem/category/name
            out = []
            for rec in data:
                if "name" not in rec:
                    continue
                # відсутнє поле dem? спробуємо витягнути з початку імені до першого "_"
                if "dem" not in rec or not rec["dem"]:
                    nm = rec["name"]
                    dem_guess = nm.split("_")[0]
                    rec["dem"] = dem_guess
                if "category" not in rec or not rec["category"]:
                    rec["category"] = _infer_category_from_name(rec["name"])
                out.append(rec)
            return out
    print("[sidebar_metadata_auto] metadata NOT found, returning empty list.")
    return []


def _infer_category_from_name(name: str) -> str:
    """Very fallback inference when category missing."""
    n = name.lower()
    if "flood" in n:
        return "flood_scenarios"
    if "hand" in n:
        # differentiate? leave base
        return "hand"
    if n.endswith("_dem"):
        return "dem"
    if "lulc" in n:
        return "lulc"
    return "unknown"


# ---------------------------------------------------------------------------
# GROUP SCHEMA
# ---------------------------------------------------------------------------
GROUP_LABEL = {
    "relief_basic": "Рельєф: базові (Aspect, Slope, Curvature)",
    "morphometric": "Морфометричні індекси (TPI, TRI, Roughness, Geomorphons)",
    "hydro_indices": "Гідрологічні індекси (TWI, Dist2Stream, HAND*)",
    "flood": "Паводки / Затоплення",
    "hydro_flow": "Гідромережа / Стік (D8, Streams)",
    "dem_prep": "Гідро-коригований DEM (Breached)",
}

CATEGORY_TO_GROUP = {
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
    # Hydro indices
    "twi": "hydro_indices",
    "distance_to_stream": "hydro_indices",
    "hand": "hydro_indices",
    "hand_500": "hydro_indices",
    "hand_1000": "hydro_indices",
    "hand_1500": "hydro_indices",
    "hand_2000": "hydro_indices",
    # Flood
    "flood_scenarios": "flood",
    # Hydro flow
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

# короткі ярлики для елементів
CAT_LABEL = {
    "aspect": "Aspect",
    "slope_horn": "Slope (Horn)",
    "slope_whitebox": "Slope (WB)",
    "curvature": "Curvature",
    "tpi": "TPI",
    "tri": "TRI",
    "roughness": "Roughness",
    "geomorphons": "Geomorphons",
    "twi": "TWI",
    "distance_to_stream": "Dist. to stream",
    "hand": "HAND",
    "hand_500": "HAND 500m",
    "hand_1000": "HAND 1000m",
    "hand_1500": "HAND 1500m",
    "hand_2000": "HAND 2000m",
    "flood_scenarios": "Flood scenarios",
    "d8_accum": "Flow accum (D8)",
    "d8_pointer": "Flow dir (D8)",
    "stream_raster": "Streams",
    "stream_raster_500": "Streams ≥500",
    "stream_raster_1000": "Streams ≥1000",
    "stream_raster_1500": "Streams ≥1500",
    "stream_raster_2000": "Streams ≥2000",
    "breached": "Hydro-conditioned DEM",
    "dem": "DEM",
    "lulc": "LULC",
    "unknown": "Unknown",
}


def cat_to_label(cat: str) -> str:
    return CAT_LABEL.get(cat, cat)


# ---------------------------------------------------------------------------
# FLOOD depth parser
# ---------------------------------------------------------------------------
_FLOOD_RE = re.compile(r"_flood_(\d+)m", flags=re.I)

def parse_flood_depth(record_or_name: Any) -> Optional[int]:
    """Return flood depth in meters or None."""
    if isinstance(record_or_name, dict):
        name = record_or_name.get("name", "")
    else:
        name = str(record_or_name)
    m = _FLOOD_RE.search(name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# GROUPED OPTIONS PER DEM
# ---------------------------------------------------------------------------
def build_grouped_records_for_dem(dem_name: str, metadata: List[Dict[str, Any]]):
    """Return dict[group_id] -> list of record dicts for that DEM."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for rec in metadata:
        if rec.get("dem") != dem_name:
            continue
        cat = rec.get("category")
        if cat in ("dem", "lulc"):
            continue
        gid = CATEGORY_TO_GROUP.get(cat)
        if not gid:
            continue
        out.setdefault(gid, []).append(rec)
    return out


def build_options_for_group(dem_name: str, gid: str, metadata: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Return dropdown options (label/value) for a group & dem."""
    grouped = build_grouped_records_for_dem(dem_name, metadata)
    recs = grouped.get(gid, [])
    return [{"label": cat_to_label(r["category"]), "value": r["name"]} for r in recs]


def build_lulc_options(metadata: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """LULC options (all available)."""
    recs = [m for m in metadata if m.get("category") == "lulc"]
    # покажемо рік, якщо є
    opts = []
    for r in recs:
        lbl = r.get("label") or r.get("year") or r["name"]
        opts.append({"label": str(lbl), "value": r["name"]})
    return opts


# ---------------------------------------------------------------------------
# SIDEBAR COMPONENT
# ---------------------------------------------------------------------------
def make_sidebar_auto() -> html.Div:
    """
    Статичний сайдбар. Усі контролі присутні постійно (щоб callback'и не ламались).
    Ми просто керуємо їхнім `style={"display":"none"}` залежно від вибраної групи.
    """
    metadata = get_metadata()
    dem_recs = [m for m in metadata if m["category"] == "dem"]
    dem_opts = [{"label": m.get("label", m["name"].upper()), "value": m["name"]} for m in dem_recs]
    default_dem = dem_opts[0]["value"] if dem_opts else None

    group_opts = [{"label": lbl, "value": gid} for gid, lbl in GROUP_LABEL.items()]

    lulc_opts = build_lulc_options(metadata)

    dropdown_style = {"width": "100%", "color": "#000"}  # текст чорний у полі, фон CSS темний

    return html.Div(
        [
            html.H4("⚙️ Фільтри", style={"color": "#FFF", "marginTop": "0.5rem"}),
            html.Label("Базовий DEM:", style={"color": "#DDD", "marginTop": "0.75rem"}),
            dcc.Dropdown(
                id="dem-select",
                options=dem_opts,
                value=default_dem,
                clearable=False,
                style=dropdown_style,
            ),

            html.Label("Похідні (оберіть тематичну групу):", style={"color": "#DDD", "marginTop": "1rem"}),
            dcc.Dropdown(
                id="derived-group",
                options=group_opts,
                placeholder="Група...",
                clearable=True,
                style=dropdown_style,
            ),

            # Контейнер другого рівня (опції всередині групи)
            html.Div(id="derived-group-controls", style={"marginTop": "0.5rem"}),

            # --- ВСІ ГРУПОВІ КОМПОНЕНТИ ЗАЗДАЛЕГІДЬ (щоб Input існували) ---
            # Заховані / показуються з callback'у render_group_controls
            dcc.Dropdown(
                id="derived-layer-multi",
                options=[],  # наповнюємо динамічно
                multi=True,
                style={"display": "none", **dropdown_style},
            ),
            dcc.Checklist(
                id="flood-visible",
                options=[{"label": "Показати затоплення", "value": "show"}],
                value=[],
                style={"display": "none", "color": "#DDD"},
            ),
            dcc.RangeSlider(
                id="flood-depth-range",
                min=1,
                max=10,
                step=1,
                value=[1, 10],
                marks={i: f"{i}m" for i in range(1, 11)},
                tooltip={"always_visible": False},
                updatemode="mouseup",
                className="flood-range-hidden",  # CSS ховає доки не активна
            ),

            html.Label("LULC:", style={"color": "#DDD", "marginTop": "1.25rem"}),
            dcc.Dropdown(
                id="lulc-select",
                options=lulc_opts,
                placeholder="Select...",
                multi=True,
                clearable=True,
                style=dropdown_style,
            ),

            # STORES --------------------------------------------------------
            dcc.Store(id="metadata-store", data=metadata),
            dcc.Store(id="selected-derived-store"),
            dcc.Store(id="selected-flood-store"),
        ],
        style={
            "width": "100%",
            "height": "100%",
            "padding": "1rem",
            "background": "#2E3B4E",
            "overflowY": "auto",
        },
    )
