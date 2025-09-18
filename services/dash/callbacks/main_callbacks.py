# callbacks/main_callbacks.py
from __future__ import annotations

import io
import logging
import os
from typing import Any, Iterable, Optional, Tuple

import dash
import duckdb
import pandas as pd
from dash import Input, Output, State, callback, dcc, html

from layout.main_tab import render_main_tab
from layout.tracks_map_tab import tracks_map_layout
from layout.tracks_profile_tab import profile_tab_layout
from registry import registry as R
from utils.plots import build_dem_stats_bar, build_error_box, build_error_hist
from utils.table import format_selected_filters, get_filtered_table_title

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger("geoai.callbacks.main")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(_h)
logger.setLevel(LOG_LEVEL)

db = R.db("nmad")

DEM_LIST: Tuple[str, ...] = (
    "alos_dem", "aster_dem", "copernicus_dem",
    "fab_dem", "nasa_dem", "srtm_dem", "tan_dem",
)
DEM_STATS_COLUMNS = [{"name": k, "id": k} for k in ["DEM", "N_points", "MAE", "RMSE", "Bias"]]

def _hand_range_effective(toggle: Optional[Iterable[str]], hand_range: Any) -> Optional[Any]:
    return hand_range if (toggle and ("on" in toggle)) else None

def _log_filters(dem, lulc, landform, slope, hand_range_) -> None:
    logger.info(
        "apply filters | dem=%s lulc=%s landform=%s slope=%s hand=%s",
        dem, lulc if lulc else "-", landform if landform else "-",
        slope if slope else "-", hand_range_ if hand_range_ is not None else "-",
    )

# --- Dependent dropdowns ---
@callback(
    Output("lulc_select", "options"),
    Output("landform_select", "options"),
    Input("dem_select", "value"),
)
def update_dropdowns(dem: str):
    try:
        lulc_opts = db.get_unique_lulc_names(dem) or []
        landform_opts = db.get_unique_landform(dem) or []
        return lulc_opts, landform_opts
    except Exception as e:
        logger.exception("failed to update dropdowns: %s", e)
        return [], []

# --- Lazy-load CDF store for tab-5 ---
@callback(
    Output("cdf-store", "data"),
    Input("idx-tabs", "value"),        # ← новий id
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

# --- Main controller: рендеримо ВСІ вкладки сюди ---
@callback(
    Output("idx-tab-content", "children"),   # ← новий контейнер
    Input("apply_filters_btn", "n_clicks"),
    Input("idx-tabs", "value"),              # ← новий id
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

    # ---- Tab 1: initial (без натискання кнопки) ----
    if tab == "tab-1" and (n_clicks is None or n_clicks == 0):
        try:
            initial_df = R.df("initial_sample")
            initial_stats_records = R.df("initial_stats").to_dict("records")

            hist_fig = build_error_hist(initial_df, dem, width=260, height=270)
            box_fig = build_error_box(initial_df, dem, width=260, height=270)
            filtered_bar = build_dem_stats_bar(initial_stats_records, width=420, height=270)

            filtered_table_title = get_filtered_table_title(lulc, landform, slope, hand_range_)
            filters_summary = format_selected_filters(lulc, landform, slope, hand_range_)
            logger.info("render tab-1 initial | dem=%s", dem)

            body = render_main_tab(
                hist_fig, box_fig, filtered_bar,
                initial_stats_records, DEM_STATS_COLUMNS,
                filtered_table_title, dem, filters_summary,
            )
            return html.Div([html.H1("DEM Comparison"), body])
        except Exception as e:
            logger.exception("tab-1 initial load failed: %s", e)
            return html.Div("Помилка початкового завантаження.")

    # ---- Tab 1: with filters ----
    if tab == "tab-1":
        _log_filters(dem, lulc, landform, slope, hand_range_)
        try:
            with duckdb.connect() as con:
                sample_df = db.get_filtered_sample(
                    con, dem, slope_range=slope, hand_range=hand_range_,
                    lulc=lulc, landform=landform, sample_n=20_000,
                )
                filtered_stats_all_dems = []
                for d in DEM_LIST:
                    s = db.get_filtered_stats(
                        con, d, slope_range=slope, hand_range=hand_range_,
                        lulc=lulc, landform=landform,
                    )
                    if s:
                        s["DEM"] = s["DEM"].replace("_", " ").upper()
                        filtered_stats_all_dems.append(s)

            filtered_bar = build_dem_stats_bar(filtered_stats_all_dems, width=420, height=270)
            hist_fig = build_error_hist(sample_df, dem, width=260, height=270)
            box_fig = build_error_box(sample_df, dem, width=260, height=270)

            filtered_table_title = get_filtered_table_title(lulc, landform, slope, hand_range_)
            filters_summary = format_selected_filters(lulc, landform, slope, hand_range_)

            body = render_main_tab(
                hist_fig, box_fig, filtered_bar,
                filtered_stats_all_dems, DEM_STATS_COLUMNS,
                filtered_table_title, dem, filters_summary,
            )
            return html.Div([html.H1("DEM Comparison"), body])
        except Exception as e:
            logger.exception("tab-1 filtered load failed: %s", e)
            return html.Div("Не вдалося застосувати фільтри.")

    # ---- Tab 2: profile ----
    if tab == "tab-2":
        logger.info("render tab-2 profile")
        return profile_tab_layout

    # ---- Tab 3: tracks map ----
    if tab == "tab-3":
        logger.info("render tab-3 map")
        return tracks_map_layout

    # ---- Tab 4: best model summaries ----
    if tab == "tab-4":
        logger.info("render tab-4 best model")
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

    # ---- Tab 5: CDF ----
    if tab == "tab-5":
        if cdf_data is None:
            logger.info("render tab-5 cdf | loading…")
            return html.Div("Завантаження CDF...")
        try:
            cdf_df = pd.read_json(io.StringIO(cdf_data), orient="split")
            from callbacks.cdf_callback import get_cdf_tab
            return get_cdf_tab(cdf_df)
        except Exception as e:
            logger.exception("failed to parse cdf json: %s", e)
            return html.Div("Не вдалося завантажити дані CDF.")

    return html.Div("Невідома вкладка")
