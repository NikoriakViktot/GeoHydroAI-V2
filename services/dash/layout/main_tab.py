# main_tab.py

from dash import html, dcc, dash_table
from utils.style import dark_table_style
import dash_bootstrap_components as dbc


def render_main_tab(
    hist_fig,
    box_fig,
    filtered_bar,
    filtered_stats_all_dems,
    columns,
    filtered_table_title,
    dem,
    filters_summary,
):
    return html.Div([
        html.H4("DEM error dashboard + Floodplain (HAND) analysis",
                style={"marginBottom": "18px"}),

        html.Div(
            'The data in the table and charts reflect only points with land cover class "Ground" '
            'according to the ATL08 product (ICESat-2).',
            style={
                "color": "#b4e0ff", "backgroundColor": "#192838",
                "marginBottom": "18px", "fontSize": "16px",
                "padding": "8px 20px", "borderRadius": "8px", "maxWidth": "820px",
            },
        ),

        # ── KPI cards grid (адаптивна сітка: 1/2/3 колонки залежно від ширини) ──
        html.Div([
            html.Div([
                html.H5([
                    f"Histogram of errors for {dem.upper().replace('_', ' ')}",
                    html.Span("ⓘ", id="hist-help",
                              style={"color": "#61dafb", "marginLeft": "8px", "cursor": "pointer"}),
                ], style={"marginBottom": "10px", "marginLeft": "10px"}),
                dbc.Tooltip(
                    "Displayed is a random sample (max 20,000 points). For illustration only.",
                    target="hist-help",
                ),
                dcc.Graph(figure=hist_fig, style={"height": "240px", "width": "100%"}),
            ], className="kpi-card"),

            html.Div([
                html.H5([
                    f"Boxplot of errors for {dem.upper().replace('_', ' ')}",
                    html.Span("ⓘ", id="box-help",
                              style={"color": "#61dafb", "marginLeft": "8px", "cursor": "pointer"}),
                ], style={"marginBottom": "10px", "marginLeft": "10px"}),
                dbc.Tooltip(
                    "Displayed is a random sample (max 20,000 points). For illustration only.",
                    target="box-help",
                ),
                dcc.Graph(figure=box_fig, style={"height": "240px", "width": "100%"}),
            ], className="kpi-card"),

            html.Div([
                html.H5("DEM performance (filtered)",
                        style={"marginBottom": "10px", "marginLeft": "10px"}),
                dcc.Graph(figure=filtered_bar, style={"height": "240px", "width": "100%"}),
            ], className="kpi-card"),
        ], className="kpi-grid", style={"marginBottom": "24px"}),

        # ── Таблиця з метриками ──
        html.Div([
            html.Div([
                html.H4(filtered_table_title),
                html.P(
                    filters_summary,
                    style={"color": "#ccc", "fontSize": "14px",
                           "marginTop": "-8px", "marginBottom": "10px"},
                ),
                dash_table.DataTable(
                    data=filtered_stats_all_dems,
                    columns=columns,
                    **dark_table_style,
                ),
            ], style={"flex": "1"}),
        ], style={"display": "flex", "gap": "28px", "marginTop": "20px"}),
    ])
