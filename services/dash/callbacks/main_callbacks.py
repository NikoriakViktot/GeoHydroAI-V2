import dash
import duckdb
from dash import html, dcc, Output, Input, State, callback
import pandas as pd
import io

from utils.plots import build_error_hist, build_error_box, build_dem_stats_bar, plot_cdf_nmad
from utils.table import get_filtered_table_title, format_selected_filters
from callbacks.cdf_callback import get_cdf_tab
from layout.main_tab import render_main_tab
from layout.tracks_profile_tab import profile_tab_layout
from layout.tracks_map_tab import tracks_map_layout

from settings import (
    NMAD_PARQUET, CDF_PARQUET, INITIAL_SAMPLE_PARQUET,
    INITIAL_STATS_PARQUET, STATS_ALL_PARQUET
)
from loaders import get_db_nmad, read_parquet

db = get_db_nmad(str(NMAD_PARQUET))

dem_list = ["alos_dem", "aster_dem", "copernicus_dem", "fab_dem", "nasa_dem", "srtm_dem", "tan_dem"]

def _hand_range_effective(toggle, hand_range):
    return hand_range if (toggle and ("on" in toggle)) else None

@callback(
    Output("lulc_select", "options"),
    Output("landform_select", "options"),
    Input("dem_select", "value"),
)
def update_dropdowns(dem):
    return db.get_unique_lulc_names(dem), db.get_unique_landform(dem)

@callback(
    Output("cdf-store", "data"),
    Input("tabs", "value"),
    prevent_initial_call=False,
)
def load_cdf_data(tab):
    if tab != "tab-5":
        raise dash.exceptions.PreventUpdate
    return read_parquet(str(CDF_PARQUET)).to_json(date_format="iso", orient="split")

@callback(
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
def update_dashboard(n_clicks, tab, cdf_data, dem, lulc, landform, slope, hand_toggle, hand_range):
    if n_clicks is None and tab != "tab-5":
        raise dash.exceptions.PreventUpdate

    hand_range_ = _hand_range_effective(hand_toggle, hand_range)

    if tab == "tab-1" and (n_clicks is None or n_clicks == 0):
        initial_df = read_parquet(str(INITIAL_SAMPLE_PARQUET))
        initial_stats = read_parquet(str(INITIAL_STATS_PARQUET)).to_dict("records")
        dem_stats_columns = [{"name": k, "id": k} for k in ["DEM", "N_points", "MAE", "RMSE", "Bias"]]

        hist_fig = build_error_hist(initial_df, dem, width=260, height=270)
        box_fig  = build_error_box(initial_df,  dem, width=260, height=270)
        filtered_bar = build_dem_stats_bar(initial_stats, width=420, height=270)

        filtered_table_title = get_filtered_table_title(lulc, landform, slope, hand_range_)
        filters_summary = format_selected_filters(lulc, landform, slope, hand_range_)

        return render_main_tab(
            hist_fig, box_fig, filtered_bar,
            initial_stats, dem_stats_columns,
            filtered_table_title, dem, filters_summary
        )

    if tab == "tab-1":
        with duckdb.connect() as con:
            sample_df = db.get_filtered_sample(
                con, dem,
                slope_range=slope, hand_range=hand_range_,
                lulc=lulc, landform=landform, sample_n=20_000
            )

            filtered_stats_all_dems_cur = []
            for d in dem_list:
                s = db.get_filtered_stats(
                    con, d,
                    slope_range=slope,
                    hand_range=hand_range_,
                    lulc=lulc,
                    landform=landform
                )
                if s:
                    s["DEM"] = s["DEM"].replace("_", " ").upper()
                    filtered_stats_all_dems_cur.append(s)

            columns = [{"name": k, "id": k} for k in ["DEM", "N_points", "MAE", "RMSE", "Bias"]]
            filtered_bar = build_dem_stats_bar(filtered_stats_all_dems_cur, width=420, height=270)

            hist_fig = build_error_hist(sample_df, dem, width=260, height=270)
            box_fig  = build_error_box(sample_df,  dem, width=260, height=270)
            filtered_table_title = get_filtered_table_title(lulc, landform, slope, hand_range_)
            filters_summary = format_selected_filters(lulc, landform, slope, hand_range_)

        return render_main_tab(
            hist_fig, box_fig, filtered_bar,
            filtered_stats_all_dems_cur, columns,
            filtered_table_title, dem, filters_summary
        )

    elif tab == "tab-2":
        return profile_tab_layout
    elif tab == "tab-3":
        return tracks_map_layout
    elif tab == "tab-4":
        return html.Div([
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
        ])
    elif tab == "tab-5":
        if cdf_data is None:
            return html.Div("Завантаження CDF...")
        try:
            cdf_df = pd.read_json(io.StringIO(cdf_data), orient="split")
        except Exception:
            return html.Div("Не вдалося завантажити дані CDF.")
        return get_cdf_tab(cdf_df)
    else:
        return html.Div("Невідома вкладка")
