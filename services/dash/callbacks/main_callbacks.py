# callbacks/main_callbacks.py
from __future__ import annotations

import io
import logging
import os
from typing import Any, Iterable, Optional, Tuple, List

import dash
import duckdb
import pandas as pd
from dash import Input, Output, State, callback, dcc, html

from callbacks.cdf_callback import get_cdf_tab
from layout.main_tab import render_main_tab
from layout.tracks_map_tab import tracks_map_layout
from layout.tracks_profile_tab import profile_tab_layout
from registry import registry as R
from utils.plots import build_dem_stats_bar, build_error_box, build_error_hist
from utils.table import format_selected_filters, get_filtered_table_title
from utils.style import apply_dark_theme

# ---------- logging ----------
_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
_LOG_LEVEL = getattr(logging, _level_name, logging.INFO)

logger = logging.getLogger("geoai.callbacks.main")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(_h)
logger.setLevel(_LOG_LEVEL)

# ---------- registry / constants ----------
db = R.db("nmad")

DEM_LIST: Tuple[str, ...] = (
    "alos_dem", "aster_dem", "copernicus_dem", "fab_dem", "nasa_dem", "srtm_dem", "tan_dem",
)
DEM_STATS_COLUMNS = [{"name": k, "id": k} for k in ["DEM", "N_points", "MAE", "RMSE", "Bias"]]

# ---------- helpers ----------
def _hand_range_effective(toggle: Optional[Iterable[str]], hand_range: Any) -> Optional[Any]:
    """Return hand_range if toggle contains 'on', otherwise None."""
    return hand_range if (toggle and ("on" in toggle)) else None


def _log_filters(dem, lulc, landform, slope, hand_range_) -> None:
    logger.info(
        "apply filters | dem=%s lulc=%s landform=%s slope=%s hand=%s",
        dem,
        lulc if lulc else "-",
        landform if landform else "-",
        slope if slope else "-",
        hand_range_ if hand_range_ is not None else "-",
    )


def _to_stats_dict(s, dem_name: str) -> Optional[dict]:
    """Normalize get_filtered_stats() result to dict or return None."""
    if s is None:
        return None
    if isinstance(s, pd.DataFrame):
        if s.empty:
            return None
        s = s.iloc[0].to_dict()
    elif isinstance(s, pd.Series):
        s = s.to_dict()
    elif not isinstance(s, dict):
        return None

    label = s.get("DEM") or dem_name
    s["DEM"] = label.replace("_", " ").upper()
    return s


def _len_df(df: Optional[pd.DataFrame]) -> int:
    return 0 if df is None else len(df)


def _empty_fig(text: str = "No data", height: int = 270):
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor="#181818",
        plot_bgcolor="#181818",
        font_color="#EEEEEE",
        height=height,
        margin=dict(l=45, r=30, t=30, b=30),
        xaxis=dict(visible=False, showgrid=False, zeroline=False),
        yaxis=dict(visible=False, showgrid=False, zeroline=False),
    )
    fig.add_annotation(
        text=text, x=0.5, y=0.5, xref="paper", yref="paper",
        showarrow=False, font=dict(size=14, color="#AAA"),
        xanchor="center", yanchor="middle",
    )
    return fig

# ---------- dropdowns ----------
@callback(
    Output("lulc_select", "options"),
    Output("landform_select", "options"),
    Input("dem_select", "value"),
)
def update_dropdowns(dem: str):
    """Populate dependent dropdowns based on DEM."""
    try:
        lulc_opts = db.get_unique_lulc_names(dem) or []
        landform_opts = db.get_unique_landform(dem) or []
        return lulc_opts, landform_opts
    except Exception as e:
        logger.exception("failed to update dropdowns: %s", e)
        return [], []

# ---------- CDF lazy-load ----------
@callback(
    Output("cdf-store", "data"),
    Input("idx-tabs", "value"),
    prevent_initial_call=False,
)
def load_cdf_data(tab: str):
    if tab != "tab-5":
        raise dash.exceptions.PreventUpdate
    try:
        cdf_df = R.df("cdf")
        return cdf_df.to_json(date_format="iso", orient="split")
    except Exception as e:
        logger.exception("failed to load cdf parquet: %s", e)
        return pd.DataFrame().to_json(date_format="iso", orient="split")

# ---------- main controller (all tabs) ----------
@callback(
    Output("idx-tab-content", "children"),
    Input("apply_filters_btn", "n_clicks"),
    Input("idx-tabs", "value"),
    Input("cdf-store", "data"),
    State("dem_select", "value"),
    State("lulc_select", "value"),
    State("landform_select", "value"),
    State("slope_slider", "value"),
    State("hand_toggle", "value"),
    State("hand_slider", "value"),
)
def update_dashboard(
    n_clicks: Optional[int],
    tab: str,
    cdf_data: Optional[str],
    dem: str,
    lulc: Optional[Iterable[str]],
    landform: Optional[Iterable[str]],
    slope: Optional[Tuple[float, float]],
    hand_toggle: Optional[Iterable[str]],
    hand_range: Optional[Tuple[float, float]],
):
    hand_range_ = _hand_range_effective(hand_toggle, hand_range)

    # ---- TAB 1: initial render (no filters applied yet) ----
    if tab == "tab-1" and (n_clicks is None or n_clicks == 0):
        try:
            initial_df = R.df("initial_sample")
            initial_stats_records = R.df("initial_stats").to_dict("records")

            if initial_df is None or (hasattr(initial_df, "empty") and initial_df.empty):
                hist_fig = _empty_fig("No sample points yet")
                box_fig = _empty_fig("No sample points yet")
            else:
                hist_fig = build_error_hist(initial_df, dem, width=260, height=270)
                box_fig  = build_error_box(initial_df, dem, width=260, height=270)

            filtered_bar = build_dem_stats_bar(initial_stats_records or [], width=420, height=270)

            for f in (hist_fig, box_fig, filtered_bar):
                apply_dark_theme(f)

            filtered_table_title = get_filtered_table_title(lulc, landform, slope, hand_range_)
            filters_summary = format_selected_filters(lulc, landform, slope, hand_range_)
            logger.info("render tab-1 initial | dem=%s", dem)

            body = render_main_tab(
                hist_fig, box_fig, filtered_bar,
                initial_stats_records or [], DEM_STATS_COLUMNS,
                filtered_table_title, dem, filters_summary,
            )
            return html.Div([html.H1("DEM Comparison"), body])
        except Exception as e:
            logger.exception("tab-1 initial load failed: %s", e)
            return html.Div("Помилка початкового завантаження.")

    # ---- TAB 1: apply filters ----
    if tab == "tab-1":
        _log_filters(dem, lulc, landform, slope, hand_range_)
        try:
            with duckdb.connect() as con:
                # sample for active DEM
                sample_df = db.get_filtered_sample(
                    con,
                    dem,
                    slope_range=slope,
                    hand_range=hand_range_,
                    lulc=lulc,
                    landform=landform,
                    sample_n=20_000,
                )

                # stats for all DEMs under current filters
                filtered_stats_all_dems: List[dict] = []
                for d in DEM_LIST:
                    raw = db.get_filtered_stats(
                        con,
                        d,
                        slope_range=slope,
                        hand_range=hand_range_,
                        lulc=lulc,
                        landform=landform,
                    )
                    stat_row = _to_stats_dict(raw, d)
                    if stat_row is not None:
                        filtered_stats_all_dems.append(stat_row)

            # figures (handle empty safely)
            if sample_df is None or (hasattr(sample_df, "empty") and sample_df.empty):
                hist_fig = _empty_fig("No points under current filters")
                box_fig  = _empty_fig("No points under current filters")
            else:
                hist_fig = build_error_hist(sample_df, dem, width=260, height=270)
                box_fig  = build_error_box(sample_df, dem, width=260, height=270)

            filtered_bar = build_dem_stats_bar(filtered_stats_all_dems, width=420, height=270)

            for f in (hist_fig, box_fig, filtered_bar):
                apply_dark_theme(f)

            filtered_table_title = get_filtered_table_title(lulc, landform, slope, hand_range_)
            filters_summary = format_selected_filters(lulc, landform, slope, hand_range_)

            sample_n = _len_df(sample_df)
            logger.info(
                "render tab-1 filtered | dem=%s stats=%d sample_n=%d",
                dem, len(filtered_stats_all_dems), sample_n
            )

            body = render_main_tab(
                hist_fig,
                box_fig,
                filtered_bar,
                filtered_stats_all_dems,
                DEM_STATS_COLUMNS,
                filtered_table_title,
                dem,
                filters_summary,
            )
            return html.Div([html.H1("DEM Comparison"), body])
        except Exception as e:
            logger.exception("tab-1 filtered load failed: %s", e)
            return html.Div("Не вдалося застосувати фільтри.")

    # ---- TAB 2: profile ----
    if tab == "tab-2":
        return profile_tab_layout

    # ---- TAB 3: tracks map ----
    if tab == "tab-3":
        return tracks_map_layout

    # ---- TAB 4: best model summaries (stub) ----
    if tab == "tab-4":
        return html.Div(
            [
                dcc.Dropdown(
                    id="groupby_dropdown",
                    options=[
                        {"label": "LULC", "value": "lulc"},
                        {"label": "Slope", "value": "slope_horn"},
                        {"label": "Geomorphon", "value": "geomorphon"},
                        {"label": "HAND", "value": "hand"},
                    ],
                    value="lulc",
                    clearable=False,
                    style={"width": "300px"},
                ),
                dcc.Graph(id="tab4-best-dem"),
                dcc.Graph(id="tab4-all-dem"),
            ]
        )

    # ---- TAB 5: CDF ----
    if tab == "tab-5":
        if not cdf_data:
            return html.Div("Завантаження CDF...")
        try:
            cdf_df = pd.read_json(io.StringIO(cdf_data), orient="split")
            return get_cdf_tab(cdf_df)
        except Exception as e:
            logger.exception("failed to parse cdf json: %s", e)
            return html.Div("Не вдалося завантажити дані CDF.")

    # ---- Fallback ----
    logger.warning("unknown tab requested: %s", tab)
    return html.Div("Невідома вкладка")
