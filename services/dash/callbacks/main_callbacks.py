# callbacks/main_callbacks.py
from __future__ import annotations
import io, logging, os
from typing import Any, Iterable, Optional, Tuple
import dash, duckdb, pandas as pd
from dash import Input, Output, State, callback, dcc, html
import dash

from callbacks.cdf_callback import get_cdf_tab
from layout.main_tab import render_main_tab
from layout.tracks_map_tab import tracks_map_layout
from layout.tracks_profile_tab import profile_tab_layout
from registry import registry as R
from utils.plots import build_dem_stats_bar, build_error_box, build_error_hist
from utils.table import format_selected_filters, get_filtered_table_title
from utils.style import apply_dark_theme

# ---- logging (робимо рівень безпечним) ----
_LEVELS = {"CRITICAL":50, "ERROR":40, "WARNING":30, "INFO":20, "DEBUG":10, "NOTSET":0}
level_name = (os.getenv("LOG_LEVEL", "INFO") or "INFO").upper()
LOG_LEVEL = _LEVELS.get(level_name, 20)  # fallback INFO
dem_list = [
    "alos_dem", "aster_dem", "copernicus_dem", "fab_dem",
    "nasa_dem", "srtm_dem", "tan_dem"
]

logger = logging.getLogger("geoai.callbacks.main")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(_h)
logger.setLevel(LOG_LEVEL)

app = dash.get_app()

# ---- registry/db  (не падаємо на імпорті) ----
try:
    db = R.db("nmad")
except Exception as e:
    logger.exception("registry db('nmad') failed: %s", e)
    db = None

DEM_LIST: Tuple[str, ...] = ("alos_dem","aster_dem","copernicus_dem","fab_dem","nasa_dem","srtm_dem","tan_dem")
DEM_STATS_COLUMNS = [{"name": k, "id": k} for k in ["DEM", "N_points", "MAE", "RMSE", "Bias"]]

def _hand_range_effective(toggle: Optional[Iterable[str]], hand_range: Any) -> Optional[Any]:
    return hand_range if (toggle and ("on" in toggle)) else None

def _log_filters(dem, lulc, landform, slope, hand_range_) -> None:
    logger.info("apply filters | dem=%s lulc=%s landform=%s slope=%s hand=%s",
                dem, lulc if lulc else "-", landform if landform else "-", slope if slope else "-",
                hand_range_ if hand_range_ is not None else "-")

def _to_stats_dict(s, dem_name: str) -> Optional[dict]:
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
    return 0 if (df is None) else len(df)

# ---------- dropdowns ----------
@app.callback(
    Output("lulc_select", "options"),
    Output("landform_select", "options"),
    Input("dem_select", "value"),
)
def update_dropdowns(dem: str):
    if db is None:
        return [], []
    try:
        lulc_opts = db.get_unique_lulc_names(dem) or []
        landform_opts = db.get_unique_landform(dem) or []
        return lulc_opts, landform_opts
    except Exception as e:
        logger.exception("failed to update dropdowns: %s", e)
        return [], []

# ---------- CDF lazy-load ----------
@app.callback(
    Output("cdf-store", "data"),
    Input("tabs", "value"),
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

# ---------- головний контролер вкладок ----------
@app.callback(
    Output("tab-content", "children"),
    Input("apply_filters_btn", "n_clicks"),
    Input("tabs", "value"),
    Input("cdf-store", "data"),
    State("dem_select", "value"),
    State("lulc_select", "value"),
    State("landform_select", "value"),
    State("slope_slider", "value"),
    State("hand_toggle", "value"),
    State("hand_slider", "value"),
)
def update_dashboard(n_clicks, tab, cdf_data,
                     dem, lulc, landform, slope, hand_toggle, hand_range):
    if n_clicks is None and tab != "tab-5":
        raise dash.exceptions.PreventUpdate
    # hand_range_ = hand_range if "on" in hand_toggle else None

    hand_range_ = _hand_range_effective(hand_toggle, hand_range)


    # === Початкове завантаження для tab-1 ===
    if tab == "tab-1" and (n_clicks is None or n_clicks == 0):
        # Завантаження з кешу (Крок 2)
        dff_plot = R.df("initial_sample")
        filtered_stats_all_dems = R.df("initial_stats").to_dict("records")
        # stats_all = R.df("stats_all_cached").to_dict("records")
        dem_stats_columns = [{"name": k, "id": k} for k in ["DEM", "N_points", "MAE", "RMSE", "Bias"]]

        # Графіки
        hist_fig = build_error_hist(dff_plot, dem, width=260, height=270)
        box_fig = build_error_box(dff_plot, dem, width=260, height=270)
        filtered_bar = build_dem_stats_bar(filtered_stats_all_dems, width=420, height=270)

        # Заголовок таблиці
        hand_range_ = hand_range if "enable" in hand_toggle else None
        filtered_table_title = get_filtered_table_title(lulc, landform, slope, hand_range_)
        filters_summary = format_selected_filters(lulc, landform, slope, hand_range_)

        return render_main_tab(
            hist_fig, box_fig, filtered_bar,
            filtered_stats_all_dems, dem_stats_columns,
            filtered_table_title, dem, filters_summary
        )

    if tab == "tab-1":
        with duckdb.connect() as con:
            # Sample для hist/box (по активному DEM, по floodplain)
            dff_plot = db.get_filtered_sample(
                con, dem,
                slope_range=slope, hand_range=hand_range_,
                lulc=lulc, landform=landform, sample_n=20_000
            )
            # Floodplain (HAND) table
            stats_hand = []
            for d in dem_list:
                s = db.get_dem_stats_sql(con, d, hand_range=hand_range_)
                if s:
                    s['DEM'] = d
                    stats_hand.append(s)
            # All territory table
            stats_all = []
            for d in dem_list:
                s = db.get_dem_stats_sql(con, d, hand_range=None)
                if s:
                    s['DEM'] = d
                    stats_all.append(s)
            columns = [{"name": k, "id": k} for k in ["DEM", "N_points", "MAE", "RMSE", "Bias"]]

            # --- Barplot для всіх DEM по поточних фільтрах ---
            filtered_stats_all_dems = []
            for d in dem_list:
                s = db.get_filtered_stats(
                    con, d,
                    slope_range=slope,
                    hand_range=hand_range_,
                    lulc=lulc,
                    landform=landform
                )
                if s:
                    s['DEM'] = d
                    filtered_stats_all_dems.append(s)
            for d in filtered_stats_all_dems:
                if "DEM" in d:
                    d["DEM"] = d["DEM"].replace("_", " ").upper()
            filtered_bar = build_dem_stats_bar(
                filtered_stats_all_dems,
                width=420, height=270
            )
            # Графіки для активного DEM
            hist_fig = build_error_hist(dff_plot, dem, width=260, height=270)
            box_fig = build_error_box(dff_plot, dem, width=260, height=270)
            filtered_table_title = get_filtered_table_title(lulc, landform, slope, hand_range_)
            filters_summary = format_selected_filters(lulc, landform, slope, hand_range_)

        return render_main_tab(
            hist_fig, box_fig, filtered_bar,
            filtered_stats_all_dems, columns,
            filtered_table_title, dem, filters_summary)

    # TAB 2,3,4,5 — як було
    if tab == "tab-2":

        return profile_tab_layout
    # if tab == "tab-3":
    #     return tracks_map_layout
    if tab == "tab-4":
        return html.Div([
            dcc.Dropdown(
                id="groupby_dropdown",
                options=[{"label":"LULC","value":"lulc"},
                         {"label":"Slope","value":"slope_horn"},
                         {"label":"Geomorphon","value":"geomorphon"},
                         {"label":"HAND","value":"hand"}],
                value="lulc", clearable=False, style={"width":"300px"},
            ),
            dcc.Graph(id="tab4-best-dem"),
            dcc.Graph(id="tab4-all-dem"),
        ])
    if tab == "tab-5":
        if not cdf_data:
            return html.Div("Завантаження CDF...")
        try:
            cdf_df = pd.read_json(io.StringIO(cdf_data), orient="split")
            return get_cdf_tab(cdf_df)
        except Exception as e:
            logger.exception("failed to parse cdf json: %s", e)
            return html.Div("Не вдалося завантажити дані CDF.")
    return html.Div("Невідома вкладка")
