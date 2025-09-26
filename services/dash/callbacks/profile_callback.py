import geopandas as gpd
import duckdb
import os
import duckdb, numpy as np, json
from pyproj import Geod

from utils.plot_track import build_profile_figure_with_hand
from utils.style import empty_dark_figure
from dash import callback, Output, Input, State, no_update, exceptions
import dash
import pandas as pd
import logging
import config as S
from layout.tracks_profile_tab import basin_json, basin_bounds

from src.interpolation_track import (
    kalman_smooth,
    interpolate_linear,
)
from registry import get_db, get_df

logger = logging.getLogger(__name__)

app = dash.get_app()
db = get_db("nmad")  # Підключаємось до NMAD, який використовується для більшості даних
DEFAULT_DEM = os.getenv("DEFAULT_TRACK_DEM", "alos_dem")
DEM_LIST = [
    "alos_dem", "aster_dem", "copernicus_dem", "fab_dem",
    "nasa_dem", "srtm_dem", "tan_dem"
]
YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
hand_column_map = {dem: f"{dem}_2000" for dem in DEM_LIST}

# --- ШЛЯХИ ДО ДАНИХ (для DuckDB запитів) ---
NMAD_PARQUET = str(S.NMAD_PARQUET)  # Дані NMAD (висоти DEM, delta, HAND)
TRACKS_PARQUET = str(S.TRACKS_PARQUET)  # Дані координат треку (x, y)


def _color_rd_bu(delta: float, vmax: float = 20.0):
    """Визначає колір точки за значенням delta (Червоний-Білий-Синій)."""
    if delta is None or not np.isfinite(delta):
        return [200, 200, 200, 180]
    x = float(np.clip(delta / vmax, -1.0, 1.0))
    if x < 0:
        t = -x;
        r, g, b = int(255 * (1 - t)), int(255 * (1 - t)), 255
    else:
        t = x;
        r, g, b = 255, int(255 * (1 - t)), int(255 * (1 - t))
    return [r, g, b, 220]


def _query_tracks(track, rgt, spot, date, dem, hand_range=None):
    """
    Отримує дані треку, об'єднуючи координати (x, y) з TRACKS_PARQUET
    і висоти/дельти (h_dem, delta) з NMAD_PARQUET.
    """
    # Фільтр HAND застосовується до таблиці NMAD (n)
    hand_sql = ""
    if hand_range and len(hand_range) == 2:
        hand_col = f"{dem}_2000"
        hand_sql = f" AND n.{hand_col} IS NOT NULL AND n.{hand_col} BETWEEN {hand_range[0]} AND {hand_range[1]}"

    sql = f"""
    WITH t AS (
        SELECT 
            CAST(x AS DOUBLE) AS x,
            CAST(y AS DOUBLE) AS y,
            time, track, rgt, spot
        FROM read_parquet('{TRACKS_PARQUET}')
        WHERE track={track} AND rgt={rgt} AND spot={spot}
          AND DATE(time) = DATE '{date}'
    ),
    n AS (
        SELECT 
            time, track, rgt, spot,
            orthometric_height,
            h_{dem}     AS h_dem,
            delta_{dem} AS delta,
            {dem}_2000 AS {dem}_2000 -- Витягуємо HAND для фільтрації
        FROM read_parquet('{NMAD_PARQUET}')
        WHERE track={track} AND rgt={rgt} AND spot={spot}
          AND DATE(time) = DATE '{date}'
          AND atl03_cnf = 4 AND atl08_class = 1
          AND h_{dem} IS NOT NULL AND delta_{dem} IS NOT NULL
    )
    SELECT 
        t.x, t.y, n.orthometric_height, n.h_dem, n.delta, n.time
    FROM t
    INNER JOIN n USING (track, rgt, spot, time)
    WHERE 1=1
      {hand_sql}
    ORDER BY t.x
    """
    try:
        return duckdb.query(sql).to_df()
    except Exception as e:
        logger.error("DuckDB tracks+NMAD query failed: %s", e)
        return pd.DataFrame()


def _add_distance(df):
    """Обчислює кумулятивну відстань уздовж треку в метрах."""
    if df is None or df.empty:
        return df
    geod = Geod(ellps="WGS84")
    d = np.zeros(len(df))
    if len(df) > 1:
        _, _, dp = geod.inv(df["x"].to_numpy()[:-1], df["y"].to_numpy()[:-1],
                            df["x"].to_numpy()[1:], df["y"].to_numpy()[1:])
        d[1:] = dp
    df = df.copy()
    df["distance_m"] = np.cumsum(d)
    return df


def _deck_spec_from_tracks(df, basemap_style):
    """Формує специфікацію DeckGL з шарами басейну, точок та лінії треку."""
    layers = []
    # 1. КОНТУР БАСЕЙНУ
    if basin_json:
        layers.append({
            "@@type": "GeoJsonLayer", "id": "basin-outline",
            "data": basin_json, "stroked": True, "filled": False,
            "getLineColor": [0, 102, 255, 220],
            "getFillColor": [0, 0, 0, 0],
            "getLineWidth": 2.5, "lineWidthUnits": "pixels",
            "lineWidthMinPixels": 2, "parameters": {"depthTest": False}
        })

    # 2. ТОЧКИ ТРЕКУ ТА ЛІНІЯ
    if df is not None and not df.empty:
        lon, lat = df["x"].to_numpy(), df["y"].to_numpy()
        delta = df["delta"].to_numpy()  # delta тепер доступна з NMAD
        # САМПЛІНГ для оптимізації відображення
        step = max(1, len(df) // 2000)
        pts = [{"position": [float(lon[i]), float(lat[i])],
                "color": _color_rd_bu(float(delta[i]))}
               for i in range(0, len(df), step)]
        path = [[float(x), float(y)] for x, y in zip(lon, lat)]

        layers += [
            {  # ScatterplotLayer (точки)
                "@@type": "ScatterplotLayer", "id": "track-points",
                "data": pts, "pickable": True,
                "parameters": {"depthTest": False},
                "radiusUnits": "pixels",
                "getRadius": 3, "radiusMinPixels": 2, "radiusMaxPixels": 8,
                "getPosition": "@@=d.position",
                "getFillColor": "@@=d.color"
            },
            {
                "@@type": "PathLayer", "id": "track-path",
                "data": [{"path": path}],
                "getPath": "@@=d.path", "widthUnits": "pixels", "getWidth": 2,
                "getColor": [255, 200, 0, 220], "parameters": {"depthTest": False}
            }
        ]

    return json.dumps({
        "mapStyle": basemap_style,
        "controller": True,
        "initialViewState": {
            "bounds": list(basin_bounds),
            "pitch": 0, "bearing": 0, "minZoom": 7, "maxZoom": 13
        },
        "layers": layers
    })


# --- DROPDOWNS & STORE CALLBACKS (БЕЗ ЗМІН) ---

@app.callback(
    Output("track_rgt_spot_dropdown", "options"),
    Output("track_rgt_spot_dropdown", "value"),
    Input("year_dropdown", "value"),
    State("selected_profile", "data"),
)
def update_tracks_dropdown(year, selected_profile):
    options = db.get_track_dropdown_options(year)
    value = options[0]["value"] if options else None
    if selected_profile and selected_profile.get("track") in [o["value"] for o in options]:
        value = selected_profile["track"]
    return options, value


@app.callback(
    Output("date_dropdown", "options"),
    Output("date_dropdown", "value"),
    Input("track_rgt_spot_dropdown", "value"),
    State("selected_profile", "data"),
)
def update_dates_dropdown(track_rgt_spot, selected_profile):
    if not track_rgt_spot:
        return [], None
    track, rgt, spot = map(float, track_rgt_spot.split("_"))
    options = db.get_date_dropdown_options(track, rgt, spot)
    value = options[0]["value"] if options else None
    if selected_profile and selected_profile.get("date") in [o["value"] for o in options]:
        value = selected_profile["date"]
    return options, value


@callback(
    Output("selected_profile", "data"),
    Output("profile_history", "data"),
    Input("year_dropdown", "value"),
    Input("track_rgt_spot_dropdown", "value"),
    Input("date_dropdown", "value"),
    State("selected_profile", "data"),
    State("profile_history", "data"),
    prevent_initial_call=True
)
def sync_profile_to_store(year, track, date, prev_profile, history):
    dem = (prev_profile or {}).get("dem", DEFAULT_DEM)
    profile = {"year": year, "track": track, "date": date, "dem": dem}

    history = history or []
    if not prev_profile or prev_profile != profile:
        history.append(profile)
    return profile, history


# --- PROFILE GRAPH CALLBACK (БЕЗ ЗМІН) ---

@callback(
    Output("track_profile_graph", "figure"),
    Output("dem_stats", "children"),
    Input("track_rgt_spot_dropdown", "value"),
    Input("date_dropdown", "value"),
    Input("hand_slider", "value"),
    Input("hand_toggle", "value"),
    Input("interp_method", "value"),
    Input("kalman_q", "value"),
    Input("kalman_r", "value"),
    State("selected_profile", "data"),
)
def update_profile(track_rgt_spot, date,
                   hand_range, hand_toggle,
                   interp_method, kalman_q, kalman_r,
                   selected_profile):
    if not track_rgt_spot or not date:
        return empty_dark_figure(text="Виберіть трек і дату"), "No error stats"

    dem = (selected_profile or {}).get("dem") or DEFAULT_DEM

    try:
        track, rgt, spot = map(float, track_rgt_spot.split("_"))
    except Exception:
        return empty_dark_figure(text="Некоректний формат треку."), "No error stats"

    use_hand = isinstance(hand_toggle, (list, tuple, set)) and "on" in hand_toggle
    hand_q = hand_range if (use_hand and hand_range and len(hand_range) == 2
                            and all(isinstance(x, (int, float)) for x in hand_range)) else None

    df_hand = db.get_profile(track, rgt, spot, dem, date, hand_q)
    df_all = db.get_profile(track, rgt, spot, dem, date, None)

    if df_hand is not None and not df_hand.empty and "distance_m" not in df_hand:
        df_hand = _add_distance(df_hand)
    if df_all is not None and not df_all.empty and "distance_m" not in df_all:
        df_all = _add_distance(df_all)

    if (df_all is None or df_all.empty or
            f"h_{dem}" not in df_all or df_all[f"h_{dem}"].dropna().empty):
        return empty_dark_figure(text="Немає даних для побудови профілю."), "No error stats"

    interpolated_df = None
    if interp_method and interp_method not in ["none", "raw", "", None]:
        df_ice = df_all[["distance_m", "orthometric_height"]].dropna().sort_values("distance_m") \
            if "distance_m" in df_all else pd.DataFrame()
        if not df_ice.empty:
            if interp_method == "linear":
                grid = np.linspace(df_ice["distance_m"].min(), df_ice["distance_m"].max(), 300)
                interpolated_df = interpolate_linear(df_ice, grid=grid)
            elif interp_method == "kalman":
                transition_cov = 10 ** (kalman_q if kalman_q is not None else -1)
                observation_cov = kalman_r if kalman_r is not None else 0.6
                smooth_df = kalman_smooth(df_ice,
                                          transition_covariance=transition_cov,
                                          observation_covariance=observation_cov)
                interpolated_df = smooth_df[["distance_m", "kalman_smooth"]].rename(
                    columns={"kalman_smooth": "orthometric_height"}
                )

    fig = build_profile_figure_with_hand(
        df_all=df_all, df_hand=df_hand,
        dem_key=dem, use_hand=use_hand,
        interpolated_df=interpolated_df, interp_method=interp_method
    )

    stats = db.get_dem_stats(df_all, dem)
    stats_text = (f"Mean error: {stats['mean']:.2f} м, Min: {stats['min']:.2f} м, "
                  f"Max: {stats['max']:.2f} м, Points: {stats['count']}"
                  if stats else "No error stats")

    return fig, stats_text


# --- DECKGL MAP CALLBACK (ВИКОРИСТОВУЄ ОНОВЛЕНИЙ _query_tracks) ---

@app.callback(
    Output("deck-track", "spec"),
    Input("selected_profile", "data"),
    Input("hand_slider", "value"),
    Input("hand_toggle", "value"),
    Input("basemap_style", "value"),
)
def update_track_map(selected_profile, hand_range, hand_toggle, basemap_style):
    # Повертаємо лише басейн, якщо немає вибраного профілю
    if not selected_profile or not all(selected_profile.values()):
        return _deck_spec_from_tracks(None, basemap_style)

    try:
        track, rgt, spot = map(float, selected_profile["track"].split("_"))
        dem = selected_profile.get("dem")
        date = selected_profile.get("date")
    except Exception:
        return _deck_spec_from_tracks(None, basemap_style)

    use_hand = isinstance(hand_toggle, (list, tuple, set)) and "on" in hand_toggle
    hand_q = hand_range if (use_hand and hand_range and len(hand_range) == 2) else None

    # ВИКОРИСТОВУЄМО ОНОВЛЕНУ ФУНКЦІЮ ЗАПИТУ
    df = _query_tracks(track, rgt, spot, date, dem, hand_q)
    df = _add_distance(df)

    # _deck_spec_from_tracks автоматично додає точки та лінію, якщо df не порожній
    return _deck_spec_from_tracks(df, basemap_style)
