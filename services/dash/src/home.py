#layout/tabs_content.py

from dash import html, dcc, dash_table
from utils.style import tab_style, selected_tab_style, dark_table_style


content = html.Div([
    html.Div(
        children=[  # ОБОВ'ЯЗКОВО передаємо через список
            dcc.Tabs(
                id="tabs",
                value="tab-1",
                className="custom-tabs",
                children=[
                    dcc.Tab(label="📊 Порівняння DEM", value="tab-1",
                            style=tab_style, selected_style=selected_tab_style),
                    dcc.Tab(label="📈 Профіль треку", value="tab-2",
                            style=tab_style, selected_style=selected_tab_style),
                    dcc.Tab(label="🗺️ Карта", value="tab-3",
                            style=tab_style, selected_style=selected_tab_style),
                    dcc.Tab(label="📋 Таблиця", value="tab-4",
                            style=tab_style, selected_style=selected_tab_style),
                    dcc.Tab(label="CDF Accumulation", value="tab-5",
                    style = tab_style, selected_style = selected_tab_style),

]
            ),
            dcc.Store(id="cdf-store"),

            # ✅ Лише один tab-content
            dcc.Loading(
                id="main-loading",
                type="circle",
                color="#2d8cff",
                children=html.Div(id="tab-content", style={"marginTop": "20px"})
            )
        ],
        style={
            "maxWidth": "1180px",
            "margin": "0 auto 24px auto",
            "paddingTop": "16px",
            "zIndex": 10,
        }
    )
], style={
    "marginLeft": "300px",
    "padding": "0 30px 24px 30px",
    "backgroundColor": "#181818",
    "color": "#EEEEEE",
    "minHeight": "100vh"
})










# import dash
# from dash import html, dcc, callback, Output, Input
# import plotly.express as px
# import plotly.graph_objs as go
# from dash import dash_table
#
# parquet_file = "data/icesat2_dem_filtered_fixed_1.parquet"
# dem_list = [
#     "alos_dem", "aster_dem", "copernicus_dem", "fab_dem",
#     "nasa_dem", "srtm_dem", "tan_dem"
# ]
# hand_column_map = {dem: f"{dem}_2000" for dem in dem_list}
#
# # -------- DARK THEME UTILS ----------
#
# def apply_dark_theme(fig):
#     fig.update_layout(
#         paper_bgcolor="#181818",
#         plot_bgcolor="#181818",
#         font_color="#EEEEEE",
#         xaxis=dict(color="#EEEEEE"),
#         yaxis=dict(color="#EEEEEE"),
#         legend=dict(font_color="#EEEEEE"),
#     )
#     return fig
#
# dark_table_style = dict(
#     style_table={"backgroundColor": "#222"},
#     style_header={
#         "backgroundColor": "#181818",
#         "color": "#EEEEEE",
#         "fontWeight": "bold",
#         "fontSize": "16px"
#     },
#     style_cell={
#         "backgroundColor": "#222",
#         "color": "#EEEEEE",
#         "border": "1px solid #333",
#         "fontFamily": "Segoe UI, Verdana, Arial, sans-serif",
#         "fontSize": "15px",
#         "textAlign": "center",
#         "padding": "5px",
#     },
#     style_data_conditional=[
#         {
#             "if": {"row_index": 0},
#             "backgroundColor": "#323b32",
#             "color": "#d4edda",
#             "fontWeight": "bold",
#         }
#     ],
# )
#
# # -------- DB UTILS ----------
#
# def duckdb_query(sql):
#     import duckdb
#     con = duckdb.connect()
#     try:
#         return con.execute(sql).fetchdf()
#     finally:
#         con.close()
#
# def get_unique_lulc_names(parquet_file, dem):
#     sql = f"SELECT DISTINCT lulc_name FROM '{parquet_file}' WHERE delta_{dem} IS NOT NULL AND lulc_name IS NOT NULL ORDER BY lulc_name"
#     try:
#         df = duckdb_query(sql)
#         return [{"label": x, "value": x} for x in df["lulc_name"].dropna().tolist()]
#     except Exception:
#         return []
#
# def get_unique_landform(parquet_file, dem):
#     sql = f"SELECT DISTINCT {dem}_landform FROM '{parquet_file}' WHERE {dem}_landform IS NOT NULL ORDER BY {dem}_landform"
#     try:
#         df = duckdb_query(sql)
#         return [{"label": x, "value": x} for x in df[f"{dem}_landform"].dropna().tolist()]
#     except Exception:
#         return []
#
# # -------- LAYOUT ----------
#
# dash.register_page(__name__, path="/")
#
# layout = html.Div([
#     html.H3("DEM error dashboard + Floodplain (HAND) analysis", style={"color": "#EEEEEE"}),
#     dcc.Dropdown(id="dem_select", options=[{'label': dem, 'value': dem} for dem in dem_list], value="alos_dem",
#                  style={"backgroundColor": "#23272b", "color": "#EEEEEE"}),
#     dcc.Dropdown(id="lulc_select", multi=True, placeholder="LULC class",
#                  style={"backgroundColor": "#23272b", "color": "#EEEEEE"}),
#     dcc.Dropdown(id="landform_select", multi=True, placeholder="Landform class",
#                  style={"backgroundColor": "#23272b", "color": "#EEEEEE"}),
#     html.Div([
#         html.Label("Похил (Slope), градуси:", style={"color": "#EEEEEE"}),
#         dcc.RangeSlider(
#             id="slope_slider", min=0, max=45, step=1, value=[0, 45],
#             marks={i: str(i) for i in range(0, 46, 10)}
#         ),
#     ], style={"margin": "10px 0 2px 0"}),
#     dcc.Checklist(
#         id="hand_filter_toggle",
#         options=[{"label": "Фільтрувати по HAND (floodplain)", "value": "on"}],
#         value=["on"],
#         style={"margin": "8px 0 4px 0", "color": "#EEEEEE"}
#     ),
#     dcc.RangeSlider(
#         id="hand_slider", min=0, max=20, step=1, value=[0, 5],
#         marks={i: str(i) for i in range(0, 21, 5)},
#         tooltip={"placement": "bottom", "always_visible": True}
#     ),
#     # --- Два графіки поруч ---
#     html.Div([
#         dcc.Graph(id="error_hist", style={"height": "340px", "width": "32%"}),
#         dcc.Graph(id="error_plot", style={"height": "340px", "width": "32%"}),
#         dcc.Graph(id="dem_stats_bar", style={"height": "340px", "width": "32%"}),
#     ],  style={"display": "flex", "gap": "12px"}),
#     html.Div(id="stats_output", style={"color": "#EEEEEE"}),
#     # --- Дві таблиці поруч (HAND vs всі дані) ---
#     html.Div([
#         html.Div([
#             html.H4("Таблиця статистики по всіх DEM у floodplain (HAND)", style={"color": "#EEEEEE"}),
#             dash_table.DataTable(
#                 id="dem_stats_table", **dark_table_style
#             ),
#         ], style={"flex": "1", "marginRight": "20px"}),
#         html.Div([
#             html.H4("Таблиця по всій території", style={"color": "#EEEEEE"}),
#             dash_table.DataTable(
#                 id="dem_stats_table_all", **dark_table_style
#             ),
#         ], style={"flex": "1"}),
#     ], style={
#         "display": "flex", "flexDirection": "row",
#         "justifyContent": "center", "alignItems": "flex-start",
#         "gap": "28px", "marginTop": "12px",
#         "backgroundColor": "#23272b",
#         "borderRadius": "14px",
#         "padding": "10px 0"
#     }),
# ], style={
#     "maxWidth": "1200px",
#     "margin": "auto",
#     "backgroundColor": "#181818",
#     "color": "#EEEEEE",
#     "minHeight": "100vh",
#     "paddingBottom": "20px"
# })
#
#
# # -------- CALLBACKS ----------
#
# @callback(
#     Output("hand_slider", "disabled"),
#     Input("hand_filter_toggle", "value")
# )
# def toggle_hand_slider(hand_filter_toggle):
#     return "on" not in hand_filter_toggle
#
# @callback(
#     Output("lulc_select", "options"),
#     Output("landform_select", "options"),
#     Input("dem_select", "value"),
# )
# def update_dropdowns(dem):
#     lulc_options = get_unique_lulc_names(parquet_file, dem)
#     landform_options = get_unique_landform(parquet_file, dem)
#     return lulc_options, landform_options
#
# @callback(
#     Output("error_hist", "figure"),
#     Output("error_plot", "figure"),
#     Output("stats_output", "children"),
#     Output("dem_stats_table", "data"),
#     Output("dem_stats_table", "columns"),
#     Output("dem_stats_table_all", "data"),
#     Output("dem_stats_table_all", "columns"),
#     Output("dem_stats_bar", "figure"),
#     Input("dem_select", "value"),
#     Input("lulc_select", "value"),
#     Input("landform_select", "value"),
#     Input("slope_slider", "value"),
#     Input("hand_slider", "value"),
#     Input("hand_filter_toggle", "value")
# )
# def update_graph(dem, lulc, landform, slope_range, hand_range, hand_filter_toggle):
#     hand_col = hand_column_map.get(dem)
#     use_hand = "on" in hand_filter_toggle
#
#     sql_all = f"SELECT delta_{dem}, {hand_col} FROM '{parquet_file}' WHERE delta_{dem} IS NOT NULL"
#     if slope_range != [0, 45]:
#         sql_all += f" AND {dem}_slope BETWEEN {slope_range[0]} AND {slope_range[1]}"
#     if lulc:
#         lulc_str = ','.join([f"'{x}'" for x in lulc])
#         sql_all += f" AND lulc_name IN ({lulc_str})"
#     if landform:
#         landform_str = ','.join([f"'{x}'" for x in landform])
#         sql_all += f" AND {dem}_landform IN ({landform_str})"
#     if hand_col and use_hand:
#         sql_all += f" AND {hand_col} BETWEEN {hand_range[0]} AND {hand_range[1]}"
#     dff_all = duckdb_query(sql_all)
#     N = len(dff_all)
#     if N == 0:
#         return go.Figure(), go.Figure(), "<b>No data for selection</b>", [], [], [], [], go.Figure()
#     base_delta = dff_all[f"delta_{dem}"]
#     base_rms = (base_delta ** 2).mean() ** 0.5
#     base_mae = base_delta.abs().mean()
#     base_bias = base_delta.mean()
#     base_stats = f"ALL: {N} точок | RMS: {base_rms:.2f} | MAE: {base_mae:.2f} | Bias: {base_bias:.2f}"
#     dff_plot = dff_all if N <= 10000 else dff_all.sample(n=10000, random_state=42)
#
#     fig = px.box(
#         dff_plot,
#         y=f"delta_{dem}",
#         points="all",
#         title=f"Error for {dem} (Sampled {len(dff_plot)} of {N})" +
#               (f"<br>HAND: {hand_range[0]}–{hand_range[1]} м" if use_hand else " (всі точки)")
#     )
#     fig = apply_dark_theme(fig)
#     hist_fig = go.Figure([go.Histogram(
#         x=dff_plot[f"delta_{dem}"], nbinsx=40, marker_color="royalblue", opacity=0.8
#     )])
#     hist_fig = apply_dark_theme(hist_fig)
#
#     dem_stats_hand = []
#     for d in dem_list:
#         hand = hand_column_map[d]
#         sql = f"SELECT delta_{d} FROM '{parquet_file}' WHERE delta_{d} IS NOT NULL AND {hand} BETWEEN {hand_range[0]} AND {hand_range[1]}"
#         df = duckdb_query(sql)
#         vals = df[f"delta_{d}"].dropna()
#         if len(vals) == 0:
#             continue
#         dem_stats_hand.append({
#             "DEM": d,
#             "N_points": len(vals),
#             "MAE": round(vals.abs().mean(), 3),
#             "RMSE": round((vals ** 2).mean() ** 0.5, 3),
#             "Bias": round(vals.mean(), 3),
#         })
#
#     # --- ВСІ дані ---
#     dem_stats_all = []
#     for d in dem_list:
#         sql = f"SELECT delta_{d} FROM '{parquet_file}' WHERE delta_{d} IS NOT NULL"
#         df = duckdb_query(sql)
#         vals = df[f"delta_{d}"].dropna()
#         if len(vals) == 0:
#             continue
#         dem_stats_all.append({
#             "DEM": d,
#             "N_points": len(vals),
#             "MAE": round(vals.abs().mean(), 3),
#             "RMSE": round((vals ** 2).mean() ** 0.5, 3),
#             "Bias": round(vals.mean(), 3),
#         })
#
#     # Сортуємо, щоб DEM, який зараз обраний, був першим (для обох таблиць)
#     dem_stats_hand = sorted(dem_stats_hand, key=lambda x: (x["DEM"] != dem, x["DEM"]))
#     dem_stats_all = sorted(dem_stats_all, key=lambda x: (x["DEM"] != dem, x["DEM"]))
#
#     columns = [{"name": k, "id": k} for k in ["DEM", "N_points", "MAE", "RMSE", "Bias"]]
#
#     # Груповий barplot (MAE, RMSE, Bias)
#     bar_fig = go.Figure()
#     bar_fig.add_trace(go.Bar(
#         x=[d["DEM"] for d in dem_stats_hand],
#         y=[d["MAE"] for d in dem_stats_hand],
#         name="MAE",
#         marker_color="#2ca02c"
#     ))
#     bar_fig.add_trace(go.Bar(
#         x=[d["DEM"] for d in dem_stats_hand],
#         y=[d["RMSE"] for d in dem_stats_hand],
#         name="RMSE",
#         marker_color="#1f77b4"
#     ))
#     bar_fig.add_trace(go.Bar(
#         x=[d["DEM"] for d in dem_stats_hand],
#         y=[d["Bias"] for d in dem_stats_hand],
#         name="Bias",
#         marker_color="#ff7f0e"
#     ))
#     bar_fig.update_layout(
#         barmode="group",
#         title="Порівняння похибок DEM " +
#               ("у floodplain (HAND)" if use_hand else " (всі точки)"),
#         xaxis_title="DEM",
#         yaxis_title="Error (м)",
#         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
#         bargap=0.18,
#     )
#     bar_fig = apply_dark_theme(bar_fig)
#
#     return (
#         hist_fig,
#         fig,
#         base_stats,
#         dem_stats_hand,
#         columns,
#         dem_stats_all,
#         columns,
#         bar_fig
#     )
