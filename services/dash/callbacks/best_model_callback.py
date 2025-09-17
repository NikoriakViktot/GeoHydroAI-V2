#callbacks/best_model_callback.py

import duckdb
from dash import html, dcc, Output, Input, State, callback
from utils.db import DuckDBData
from utils.plots import build_best_dem_barplot, build_grouped_nmad_barplot

db = DuckDBData("data/NMAD_dem.parquet")

# Це можна винести у файл constants.py
landform_names = {
    1: "Flat", 2: "Peak", 3: "Ridge", 4: "Shoulder", 5: "Spur",
    6: "Slope", 7: "Hollow", 8: "Footslope", 9: "Valley", 10: "Pit"
}

@callback(
    Output("tab4-best-dem", "figure"),
    Output("tab4-all-dem", "figure"),
    Input("apply_filters_btn", "n_clicks"),
    Input("groupby_dropdown", "value"),
    State("dem_select", "value"),
    State("lulc_select", "value"),
    State("landform_select", "value"),
    State("slope_slider", "value"),
    State("hand_toggle", "value"),
    State("hand_slider", "value"),
    prevent_initial_call=False,
)
def update_best_dem_tab(n_clicks, groupby, dem, lulc, landform, slope, hand_toggle, hand_range):
    hand_range_ = hand_range if hand_toggle and "on" in hand_toggle else None
    with duckdb.connect() as con:
        if groupby == "lulc":
            df = db.get_nmad_grouped_by_lulc(
                con, slope_range=slope, hand_range=hand_range_, lulc=lulc, landform=landform
            )
            fig1 = build_best_dem_barplot(df, x_col="lulc_name", title="🏆 Найкраща DEM для кожного класу LULC (NMAD)")
            fig2 = build_grouped_nmad_barplot(df, x_col="lulc_name", title="NMAD для кожного DEM у класах LULC")
        elif groupby == "slope_horn":
            df = db.get_nmad_grouped_by_slope(
                con, slope_range=slope, hand_range=hand_range_, lulc=lulc, landform=landform
            )
            fig1 = build_best_dem_barplot(df, x_col="slope_class", title="🏆 Найкраща DEM для кожного класу схилу (NMAD)")
            fig2 = build_grouped_nmad_barplot(df, x_col="slope_class", title="NMAD для кожного DEM у класах схилу")
        elif groupby == "geomorphon":
            df = db.get_nmad_grouped_by_geomorphon(
                con, slope_range=slope, hand_range=hand_range_, lulc=lulc, landform=landform
            )
            fig1 = build_best_dem_barplot(df, x_col="landform", name_dict=landform_names,
                                          title="🏆 Найкраща DEM для кожного геоморфону (NMAD)")
            fig2 = build_grouped_nmad_barplot(df, x_col="landform", name_dict=landform_names,
                                              title="NMAD для кожного DEM у класах геоморфонів")
        elif groupby == "hand":
            df = db.get_nmad_grouped_by_hand(
                con, slope_range=slope, hand_range=hand_range_, lulc=lulc, landform=landform
            )
            fig1 = build_best_dem_barplot(df, x_col="hand_class", title="🏆 Найкраща DEM для кожного класу HAND (NMAD)")
            fig2 = build_grouped_nmad_barplot(df, x_col="hand_class", title="NMAD для кожного DEM у класах HAND")
        else:
            return {}, {}
    return fig1, fig2
