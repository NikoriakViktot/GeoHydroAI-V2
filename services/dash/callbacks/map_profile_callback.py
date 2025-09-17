from dash import callback, Output, Input, State
import dash_leaflet as dl
import plotly.graph_objs as go
import duckdb
import pandas as pd
import dash


from dash import callback, Output, Input, State

@callback(
    Output("tabs", "value"),
    Input("go_to_profile_btn", "n_clicks"),
    State("tabs", "value"),
    prevent_initial_call=True
)
def go_to_profile(profile_clicks, tab):
    if profile_clicks:
        return "tab-2"
    raise dash.exceptions.PreventUpdate


# DEM_LIST = [
#     "alos_dem", "aster_dem", "copernicus_dem", "fab_dem",
#     "nasa_dem", "srtm_dem", "tan_dem"
# ]
# hand_column_map = {dem: f"{dem}_2000" for dem in DEM_LIST}
#
#
# # --- (a) Дропдауни (залишаємо як є, якщо треба динамічно)
# @callback(
#     Output("track_rgt_spot_dropdown", "options"),
#     Output("track_rgt_spot_dropdown", "value"),
#     Input("year_dropdown", "value"),
# )
# def update_tracks_dropdown(year):
#     sql = f"""
#         SELECT DISTINCT track, rgt, spot
#         FROM 'data/tracks_3857_1.parquet'
#         WHERE year = {year}
#           AND atl03_cnf = 4 AND atl08_class = 1
#         ORDER BY track, rgt, spot
#     """
#     try:
#         df = duckdb.query(sql).to_df()
#         options = [
#             {"label": f"Track {row.track} / RGT {row.rgt} / Spot {row.spot}", "value": f"{row.track}_{row.rgt}_{row.spot}"}
#             for _, row in df.iterrows()
#         ]
#         value = options[0]["value"] if options else None
#         return options, value
#     except Exception:
#         return [], None
#
# @callback(
#     Output("date_dropdown", "options"),
#     Output("date_dropdown", "value"),
#     Input("track_rgt_spot_dropdown", "value"),
# )
# def update_dates_dropdown(track_rgt_spot):
#     if not track_rgt_spot:
#         return [], None
#     track, rgt, spot = map(float, track_rgt_spot.split("_"))
#     sql = f"""
#         SELECT DISTINCT DATE(time) as date_only
#         FROM 'data/tracks_3857_1.parquet'
#         WHERE track={track} AND rgt={rgt} AND spot={spot}
#             AND atl03_cnf = 4 AND atl08_class = 1
#         ORDER BY date_only
#     """
#     try:
#         df = duckdb.query(sql).to_df()
#         options = [{
#             "label": pd.to_datetime(row.date_only).strftime("%Y-%m-%d"),
#             "value": pd.to_datetime(row.date_only).strftime("%Y-%m-%d")
#         } for _, row in df.iterrows()]
#         value = options[0]["value"] if options else None
#         return options, value
#     except Exception:
#         return [], None
#
# def get_track_data_for_date(track, rgt, spot, dem, date, hand_range=None):
#     hand_col = hand_column_map[dem]
#     sql = f"""
#         SELECT *
#         FROM 'data/tracks_3857_1.parquet'
#         WHERE track={track} AND rgt={rgt} AND spot={spot}
#             AND DATE(time) = '{date}'
#             AND delta_{dem} IS NOT NULL AND h_{dem} IS NOT NULL
#             AND atl03_cnf = 4 AND atl08_class = 1
#     """
#     if hand_range and len(hand_range) == 2 and all(x is not None for x in hand_range):
#         sql += f" AND {hand_col} IS NOT NULL AND {hand_col} BETWEEN {hand_range[0]} AND {hand_range[1]}"
#     sql += " ORDER BY x"
#     try:
#         df = duckdb.query(sql).to_df()
#         return df
#     except Exception:
#         return pd.DataFrame()
#
# def get_dem_stats(df, dem_key):
#     delta_col = f"delta_{dem_key}"
#     if delta_col not in df:
#         return None
#     delta = df[delta_col].dropna()
#     if delta.empty:
#         return None
#     return {
#         "mean": delta.mean(),
#         "min": delta.min(),
#         "max": delta.max(),
#         "count": len(delta)
#     }
#
# def build_profile_figure_with_hand(df_all, df_hand, dem_key, use_hand):
#     fig = go.Figure()
#     if not df_all.empty and f"h_{dem_key}" in df_all:
#         x_axis_dem = df_all["distance_m"] if "distance_m" in df_all else df_all["x"]
#         fig.add_trace(go.Scatter(
#             x=x_axis_dem,
#             y=df_all[f"h_{dem_key}"],
#             mode="markers",
#             marker=dict(size=2, color="lightgray"),
#             name=f"{dem_key.upper()} DEM",
#             opacity=0.9
#         ))
#     show_df = df_hand if (use_hand and not df_hand.empty) else df_all
#     if not show_df.empty and "orthometric_height" in show_df:
#         x_axis_ice = show_df["distance_m"] if "distance_m" in show_df else show_df["x"]
#         fig.add_trace(go.Scatter(
#             x=x_axis_ice,
#             y=show_df["orthometric_height"],
#             mode="markers",
#             marker=dict(size=2, color="crimson"),
#             name="ICESat-2 Orthometric Height",
#             opacity=0.9
#         ))
#     if f"delta_{dem_key}" in df_all and not df_all[f"delta_{dem_key}"].dropna().empty:
#         delta = df_all[f"delta_{dem_key}"].dropna()
#         stats_text = (
#             f"Похибка {dem_key.upper()}: "
#             f"Сер: {delta.mean():.2f} м, "
#             f"Мін: {delta.min():.2f} м, "
#             f"Макс: {delta.max():.2f} м"
#         )
#         fig.add_annotation(
#             text=stats_text,
#             xref="paper", yref="paper",
#             x=0.02, y=0.99,
#             showarrow=False,
#             font=dict(size=13, color="lightgray", family="monospace"),
#             align="left",
#             bordercolor="gray", borderwidth=1,
#             xanchor="left"
#         )
#     fig.update_layout(
#         xaxis=dict(title="Відстань/Longitude", gridcolor="#666", gridwidth=0.6, griddash="dot", zerolinecolor="#555"),
#         yaxis=dict(title="Ортометрична висота (м)", gridcolor="#666", gridwidth=0.3, griddash="dot", zerolinecolor="#555"),
#         height=600,
#         legend=dict(orientation="h", y=1.06, x=0.5, xanchor="center", font=dict(size=12), bgcolor='rgba(0,0,0,0)'),
#         plot_bgcolor="#20232A",
#         paper_bgcolor="#181818",
#         font_color="#EEE",
#         margin=dict(l=70, r=50, t=40, b=40)
#     )
#     return fig
#
# # --- (d) Зберігання історії профілів і відмальовка на карті (Store-based pattern)
# @callback(
#     Output("point_group", "children"),           # Маркери на карті
#     Output("track_profile_graph", "figure"),     # Графік профілю
#     Output("dem_stats", "children"),             # Статистика DEM
#     Output("profile_history", "data"),           # Зберігаємо історію профілів
#     Input("selected_profile", "data"),
#     Input("hand_slider", "value"),
#     Input("hand_toggle", "value"),
#     State("profile_history", "data"),
# )
# def update_map_and_profile(selected_profile, hand_range, hand_toggle, history):
#     if not selected_profile or not all(selected_profile.values()):
#         return [], go.Figure(), "", history or []
#     track_str = selected_profile["track"]
#     dem = selected_profile["dem"]
#     date = selected_profile["date"]
#     try:
#         track, rgt, spot = map(float, track_str.split("_"))
#     except Exception:
#         return [], go.Figure(), "", history or []
#
#     use_hand = "on" in hand_toggle
#     hand_range_for_query = hand_range if (use_hand and hand_range and len(hand_range) == 2 and all(isinstance(x, (int, float)) for x in hand_range)) else None
#
#     # --- Дані для графіка та статистики ---
#     df_hand = get_track_data_for_date(track, rgt, spot, dem, date, hand_range_for_query)
#     df_all = get_track_data_for_date(track, rgt, spot, dem, date, None)
#     fig = build_profile_figure_with_hand(df_all, df_hand, dem, use_hand)
#     stats = get_dem_stats(df_all, dem)
#     stats_text = (
#         f"Mean error: {stats['mean']:.2f} м, Min: {stats['min']:.2f} м, Max: {stats['max']:.2f} м, Points: {stats['count']}"
#         if stats else "No error stats"
#     )
#
#     # --- Дані для карти ---
#     geojson = get_geojson_for_date(track, rgt, spot, dem, date, hand_range_for_query, step=50)
#     features = geojson["features"]
#     if not features:
#         return [], fig, stats_text, history or []
#
#     valid_features = [(i, f) for i, f in enumerate(features) if f["properties"]["delta"] is not None]
#     if not valid_features:
#         return [], fig, stats_text, history or []
#     deltas = [f["properties"]["delta"] for i, f in valid_features]
#     min_val = min(deltas)
#     max_val = max(deltas)
#     min_idx = valid_features[deltas.index(min_val)][0]
#     max_idx = valid_features[deltas.index(max_val)][0]
#
#     pts = [
#         dl.CircleMarker(
#             center=[feat["geometry"]["coordinates"][1], feat["geometry"]["coordinates"][0]],
#             radius=2, color="blue", fillColor="blue", fillOpacity=0.8,
#             children=[dl.Tooltip(f"{feat['properties']['delta']:.2f} м")]
#         )
#         for i, feat in enumerate(features) if i not in [min_idx, max_idx]
#     ]
#     feat_min = features[min_idx]
#     feat_max = features[max_idx]
#     if min_idx != max_idx:
#         pts.append(
#             dl.CircleMarker(
#                 center=[feat_min["geometry"]["coordinates"][1], feat_min["geometry"]["coordinates"][0]],
#                 radius=5, color="lime", fillColor="lime", fillOpacity=1,
#                 children=[dl.Tooltip(f"Min: {feat_min['properties']['delta']:.2f} м")]
#             )
#         )
#         pts.append(
#             dl.CircleMarker(
#                 center=[feat_max["geometry"]["coordinates"][1], feat_max["geometry"]["coordinates"][0]],
#                 radius=5, color="red", fillColor="red", fillOpacity=1,
#                 children=[dl.Tooltip(f"Max: {feat_max['properties']['delta']:.2f} м")]
#             )
#         )
#     else:
#         pts.append(
#             dl.CircleMarker(
#                 center=[feat_min["geometry"]["coordinates"][1], feat_min["geometry"]["coordinates"][0]],
#                 radius=5, color="cyan", fillColor="cyan", fillOpacity=1,
#                 children=[dl.Tooltip(f"Point: {feat_min['properties']['delta']:.2f} м")]
#             )
#         )
#
#     # --- Оновлюємо історію профілів (записуємо тільки якщо новий профіль) ---
#     if history is None:
#         history = []
#     if not history or history[-1] != selected_profile:
#         history.append(selected_profile)
#     return pts, fig, stats_text, history
