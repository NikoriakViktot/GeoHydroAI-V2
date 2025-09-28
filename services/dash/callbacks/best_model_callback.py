#callbacks/best_model_callback.py
import dash
import logging
import duckdb
from dash import html, dcc, Output, Input, State, callback
from utils.plots import build_best_dem_barplot, build_grouped_nmad_barplot
from registry import get_db
from utils.style import empty_dark_figure

db = get_db("nmad")
app = dash.get_app()

# db = DuckDBData("data/NMAD_dem.parquet")
logger = logging.getLogger("geoai.callbacks.best_model_callback")

# Це можна винести у файл constants.py
landform_names = {
    1: "Flat", 2: "Peak", 3: "Ridge", 4: "Shoulder", 5: "Spur",
    6: "Slope", 7: "Hollow", 8: "Footslope", 9: "Valley", 10: "Pit"
}

@app.callback(
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
        try:
            if groupby == "lulc":
                df = db.get_nmad_grouped_by_lulc(con, slope_range=slope,
                                                 hand_range=hand_range_, lulc=lulc, landform=landform)
                fig1 = build_best_dem_barplot(df, x_col="lulc_name",
                                              title="🏆 Best-performing DEM for each LULC class (NMAD)")
                fig2 = build_grouped_nmad_barplot(df, x_col="lulc_name",
                                                  title="NMAD of each DEM within LULC classes")

            elif groupby == "slope_horn":
                df = db.get_nmad_grouped_by_slope(con, slope_range=slope,
                                                  hand_range=hand_range_, lulc=lulc, landform=landform)
                fig1 = build_best_dem_barplot(df, x_col="slope_class",
                                              title="🏆 Best-performing DEM for each slope class (NMAD)")
                fig2 = build_grouped_nmad_barplot(df, x_col="slope_class",
                                                  title="NMAD of each DEM within slope classes")

            elif groupby == "geomorphon":
                df = db.get_nmad_grouped_by_geomorphon(con, slope_range=slope,
                                                       hand_range=hand_range_, lulc=lulc, landform=landform)
                fig1 = build_best_dem_barplot(df, x_col="landform", name_dict=landform_names,
                                              title="🏆 Best-performing DEM for each geomorphon class (NMAD)")
                fig2 = build_grouped_nmad_barplot(df, x_col="landform", name_dict=landform_names,
                                                  title="NMAD of each DEM within geomorphon classes")

            elif groupby == "hand":
                df = db.get_nmad_grouped_by_hand(con, slope_range=slope,
                                                 hand_range=hand_range_, lulc=lulc, landform=landform)
                fig1 = build_best_dem_barplot(df, x_col="hand_class",
                                              title="🏆 Best-performing DEM for each HAND class (NMAD)")
                fig2 = build_grouped_nmad_barplot(df, x_col="hand_class",
                                                  title="NMAD of each DEM within HAND classes")
            else:
                return empty_dark_figure(text="Error loading data"), empty_dark_figure(text="Error loading data")

        except Exception as e:
            logger.exception("Failed to update best DEM tab: %s", e)
            return empty_dark_figure(text="Error loading data"), empty_dark_figure(text="Error loading data")

    return fig1, fig2

