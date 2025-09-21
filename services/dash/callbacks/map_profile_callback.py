from dash import callback, Output, Input, State, no_update, exceptions
import plotly.graph_objs as go
import dash_leaflet as dl
import pandas as pd
import json

# --- НОВИЙ СПОСІБ ДОСТУПУ ДО ДАНИХ ---
from registry import get_db

app = dash.get_app()
# Отримуємо об'єкт для роботи з базою треків через registry
db = get_db("tracks")


@callback(
    Output("tabs", "value"),
    Input("go_to_profile_btn", "n_clicks"),
    prevent_initial_call=True
)
def go_to_profile(profile_clicks):
    if profile_clicks:
        return "tab-2"
    return no_update


# --- (a) Дропдауни, переписані на методи db ---
@callback(
    Output("track_rgt_spot_dropdown", "options"),
    Output("track_rgt_spot_dropdown", "value"),
    Input("year_dropdown", "value"),
)
def update_tracks_dropdown(year):
    # БУЛО: довгий SQL-запит до файлу
    # СТАЛО: виклик методу, який інкапсулює логіку
    try:
        options = db.get_track_options_for_year(year)
        value = options[0]["value"] if options else None
        return options, value
    except Exception:
        return [], None


@callback(
    Output("date_dropdown", "options"),
    Output("date_dropdown", "value"),
    Input("track_rgt_spot_dropdown", "value"),
)
def update_dates_dropdown(track_rgt_spot):
    if not track_rgt_spot:
        return [], None

    # БУЛО: довгий SQL-запит до файлу
    # СТАЛО: виклик методу
    try:
        track, rgt, spot = map(float, track_rgt_spot.split("_"))
        options = db.get_date_options_for_track(track, rgt, spot)
        value = options[0]["value"] if options else None
        return options, value
    except Exception:
        return [], None


# --- Функції-хелпери залишаються, але вони тепер викликають методи db ---
def build_profile_figure(df_profile, dem_key):
    # Ця функція може залишитись майже без змін, бо вона працює з DataFrame
    fig = go.Figure()
    if not df_profile.empty:
        fig.add_trace(go.Scatter(
            x=df_profile["distance_m"],
            y=df_profile[f"h_{dem_key}"],
            mode="markers", name=f"{dem_key.upper()} DEM",
            marker=dict(size=2, color="lightgray")
        ))
        fig.add_trace(go.Scatter(
            x=df_profile["distance_m"],
            y=df_profile["orthometric_height"],
            mode="markers", name="ICESat-2",
            marker=dict(size=2, color="crimson")
        ))
    # ... налаштування layout фігури ...
    fig.update_layout(
        plot_bgcolor="#20232A", paper_bgcolor="#181818", font_color="#EEE"
    )
    return fig


# --- (d) Головний колбек для оновлення карти та профілю ---
@callback(
    Output("point_group", "children"),
    Output("track_profile_graph", "figure"),
    Output("dem_stats", "children"),
    Input("selected_profile", "data"),  # Припускаємо, що цей store заповнюється десь ще
)
def update_map_and_profile(selected_profile):
    if not selected_profile or not all(selected_profile.values()):
        return [], go.Figure(), "No profile selected"

    dem = selected_profile["dem"]
    date = selected_profile["date"]
    track, rgt, spot = map(float, selected_profile["track"].split("_"))

    # --- Дані для графіка та статистики ---
    # БУЛО: get_track_data_for_date з SQL-запитом
    # СТАЛО: виклик методу db
    df_profile = db.get_track_profile(track, rgt, spot, dem, date)

    fig = build_profile_figure(df_profile, dem)
    stats = db.get_dem_stats(df_profile, dem)  # Цей метод теж має бути в db
    stats_text = (
        f"Mean error: {stats['mean']:.2f} м, Min: {stats['min']:.2f} м, Max: {stats['max']:.2f} м, Points: {stats['count']}"
        if stats else "No error stats"
    )

    # --- Дані для карти ---
    # БУЛО: get_geojson_for_date
    # СТАЛО: виклик методу db, який повертає GeoJSON-сумісний словник
    geojson_data = db.get_track_geojson(track, rgt, spot, dem, date)

    # Конвертація в маркери залишається такою ж, але працює з dict, а не GeoDataFrame
    markers = [
        dl.CircleMarker(
            center=[feat["geometry"]["coordinates"][1], feat["geometry"]["coordinates"][0]],
            radius=3, color="blue",
            children=[dl.Tooltip(f"Error: {feat['properties'].get(f'delta_{dem}', 'N/A'):.2f} m")]
        )
        for feat in geojson_data.get("features", [])
    ]

    return markers, fig, stats_text
