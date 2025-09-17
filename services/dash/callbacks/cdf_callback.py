# cdf_callback.py
from dash import html, dcc
from utils.plots import plot_cdf_nmad
import dash
import pandas as pd



def get_cdf_tab(cdf_df_or_json):
    if isinstance(cdf_df_or_json, str):  # JSON case
        cdf_df = pd.read_json(cdf_df_or_json, orient="split")
    else:
        cdf_df = cdf_df_or_json
    fig = plot_cdf_nmad(cdf_df)
    return html.Div([
        html.H4("CDF Accumulation Curve"),
        html.P(
            "This plot shows the cumulative distribution function (CDF) of NMAD errors for each DEM. "
            "The curve indicates the fraction of points (Y axis) with NMAD below a given threshold (X axis). "
            "Steeper curves near the origin indicate higher DEM accuracy."
        ),
        dcc.Graph(figure=fig)
    ])
