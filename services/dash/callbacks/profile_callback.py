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
from layout.tracks_map_tab import basin_json, basin_bounds

from src.interpolation_track import (
        kalman_smooth,
    interpolate_linear,
    )
from registry import get_db, get_df

logger = logging.getLogger(__name__)

app = dash.get_app()
nmad_db = get_db("nmad")
DEFAULT_DEM = os.getenv("DEFAULT_TRACK_DEM", "alos_dem")
DEM_LIST = [
    "alos_dem", "aster_dem", "copernicus_dem", "fab_dem",
    "nasa_dem", "srtm_dem", "tan_dem"
]
YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
hand_column_map = {dem: f"{dem}_2000" for dem in DEM_LIST}




# --- Track/RGT/Spot Dropdown
@callback(
    Output("track_rgt_spot_dropdown", "options"),
    Output("track_rgt_spot_dropdown", "value"),
    Input("year_dropdown", "value"),
    State("selected_profile", "data"),
)
def update_tracks_dropdown(year, selected_profile):
    options = nmad_db.get_track_dropdown_options(year)
    value = options[0]["value"] if options else None
    if selected_profile and selected_profile.get("track") in [o["value"] for o in options]:
        value = selected_profile["track"]
    return options, value

@callback(
    Output("date_dropdown", "options"),
    Output("date_dropdown", "value"),
    Input("track_rgt_spot_dropdown", "value"),
    State("selected_profile", "data"),
)
def update_dates_dropdown(track_rgt_spot, selected_profile):
    if not track_rgt_spot:
        return [], None
    track, rgt, spot = map(float, track_rgt_spot.split("_"))
    options = nmad_db.get_date_dropdown_options(track, rgt, spot)
    value = options[0]["value"] if options else None
    if selected_profile and selected_profile.get("date") in [o["value"] for o in options]:
        value = selected_profile["date"]
    return options, value


# --- STORE: єдиний callback для синхронізації state/history
@callback(
    Output("selected_profile", "data"),
    Output("profile_history", "data"),
    Input("year_dropdown", "value"),
    Input("track_rgt_spot_dropdown", "value"),
    Input("dem_select", "value"),
    Input("date_dropdown", "value"),
    State("selected_profile", "data"),
    State("profile_history", "data"),
    prevent_initial_call=True
)
def sync_profile_to_store(year, track, dem, date, prev_profile, history):
    # Записуємо весь поточний профіль
    profile = {"year": year, "track": track, "dem": dem, "date": date}
    if not history:
        history = []
    if not prev_profile or prev_profile != profile:
        history.append(profile)
    return profile, history


def add_distance_m(df, lon_col="x", lat_col="y"):
    geod = Geod(ellps="WGS84")
    if lon_col in df and lat_col in df:
        lons = df[lon_col].values
        lats = df[lat_col].values
        # Вираховуємо послідовні відстані між точками в метрах
        dists = np.zeros(len(df))
        if len(df) > 1:
            _, _, dists_pair = geod.inv(lons[:-1], lats[:-1], lons[1:], lats[1:])
            dists[1:] = dists_pair
        df = df.copy()
        df["distance_m"] = np.cumsum(dists)
    else:
        df["distance_m"] = np.arange(len(df))
    return df

from utils.style import empty_dark_figure
from pyproj import Geod
import numpy as np
import pandas as pd
import logging
logger = logging.getLogger(__name__)

@callback(
    Output("track_profile_graph", "figure"),
    Output("dem_stats", "children"),
    Input("track_rgt_spot_dropdown", "value"),
    Input("dem_select", "value"),
    Input("date_dropdown", "value"),
    Input("hand_slider", "value"),
    Input("hand_toggle", "value"),
    Input("interp_method", "value"),
    Input("kalman_q", "value"),
    Input("kalman_r", "value"),
)
def update_profile(track_rgt_spot, dem, date,
                   hand_range, hand_toggle, interp_method, kalman_q, kalman_r):
    try:
        # 1) базові перевірки
        if not (track_rgt_spot and date and dem):
            return empty_dark_figure(text="Select track/date/DEM"), "No error statistics"

        try:
            track, rgt, spot = map(float, track_rgt_spot.split("_"))
        except Exception:
            return empty_dark_figure(text="Invalid track format"), "No error statistics"

        # 2) HAND toggle безпечний
        hand_toggle = hand_toggle or []
        if isinstance(hand_toggle, str):
            hand_toggle = [hand_toggle]
        use_hand = "on" in hand_toggle

        hand_q = (hand_range if (use_hand and hand_range and len(hand_range) == 2
                                 and all(isinstance(x, (int, float)) for x in hand_range))
                  else None)

        # 3) Дані
        df_hand = nmad_db.get_profile(track, rgt, spot, dem, date, hand_q)
        df_all  = nmad_db.get_profile(track, rgt, spot, dem, date, None)

        # 4) distance_m, але не падаємо, якщо None/порожньо
        def add_distance_m(df):
            if not isinstance(df, pd.DataFrame) or df.empty:
                return df
            if "distance_m" in df:
                return df
            geod = Geod(ellps="WGS84")
            lons, lats = df["x"].to_numpy(), df["y"].to_numpy()
            d = np.zeros(len(df))
            if len(df) > 1:
                _, _, d_pair = geod.inv(lons[:-1], lats[:-1], lons[1:], lats[1:])
                d[1:] = d_pair
            out = df.copy()
            out["distance_m"] = np.cumsum(d)
            return out

        df_hand = add_distance_m(df_hand)
        df_all  = add_distance_m(df_all)

        # 5) якщо взагалі немає DEM або воно порожнє — повертаємо пустий граф
        if (not isinstance(df_all, pd.DataFrame) or df_all.empty or
                f"h_{dem}" not in df_all or df_all[f"h_{dem}"].dropna().empty):
            return empty_dark_figure(text="No profile data available"), "No error statistics"

        # 6) Інтерполяція з безпечними дефолтами
        interpolated_df = None
        if interp_method and interp_method not in ["none", "raw", "", None]:
            if "distance_m" in df_all and "orthometric_height" in df_all:
                df_ice = df_all[["distance_m", "orthometric_height"]].dropna().sort_values("distance_m")
                if not df_ice.empty:
                    if interp_method == "linear":
                        grid = np.linspace(df_ice["distance_m"].min(), df_ice["distance_m"].max(), 300)
                        interpolated_df = interpolate_linear(df_ice, grid=grid)

                    elif interp_method == "kalman":
                        kq = kalman_q if isinstance(kalman_q, (int, float)) else -1  # 10**-1 = 0.1
                        kr = kalman_r if isinstance(kalman_r, (int, float)) else 0.6
                        smooth_df = kalman_smooth(
                            df_ice,
                            transition_covariance=10 ** kq,  # Q_base
                            observation_covariance=kr,  # R
                            gap_break=180.0
                        )
                        interpolated_df = smooth_df[["distance_m", "kalman_smooth"]].rename(
                            columns={"kalman_smooth": "orthometric_height"}
                        )
                    # elif interp_method == "kalman":
                    #     kq = kalman_q if isinstance(kalman_q, (int, float)) else -1  # 10**-1 = 0.1
                    #     kr = kalman_r if isinstance(kalman_r, (int, float)) else 0.6
                    #     smooth_df = kalman_smooth(
                    #         df_ice,
                    #         transition_covariance=10 ** kq,
                    #         observation_covariance=kr
                    #     )
                    #     interpolated_df = smooth_df[["distance_m", "kalman_smooth"]].rename(
                    #         columns={"kalman_smooth": "orthometric_height"}
                    #     )

        # 7) Малюємо (твоя функція — з п. A)
        fig = build_profile_figure_with_hand(
            df_all=df_all,
            df_hand=df_hand,
            dem_key=dem,
            use_hand=use_hand,
            interpolated_df=interpolated_df,
            interp_method=interp_method,
        )

        # 8) Статистика
        stats = nmad_db.get_dem_stats(df_all, dem)
        stats_text = (
            f"Mean error: {stats['mean']:.2f} м, Min: {stats['min']:.2f} м, "
            f"Max: {stats['max']:.2f} м, Points: {stats['count']}"
            if stats else "No error stats"
        )
        return fig, stats_text

    except Exception as e:
        logger.exception("update_profile failed")
        return empty_dark_figure(text=f"Server error: {e}"), "No error stats"



# TRACKS_PARQUET = str(S.TRACKS_PARQUET)
#
# def _color_rd_bu(delta: float, vmax: float = 20.0):
#     if delta is None or not np.isfinite(delta):
#         return [200, 200, 200, 180]
#     x = float(np.clip(delta / vmax, -1.0, 1.0))
#     if x < 0:
#         t = -x;   r, g, b = int(255*(1-t)), int(255*(1-t)), 255
#     else:
#         t = x;    r, g, b = 255, int(255*(1-t)), int(255*(1-t))
#     return [r, g, b, 220]
#
# def _query_tracks(track, rgt, spot, date, dem, hand_range=None):
#     # фільтр HAND (колонка типу alos_dem_2000)
#     hand_sql = ""
#     if hand_range and len(hand_range) == 2:
#         hand_col = f"{dem}_2000"
#         hand_sql = f" AND {hand_col} IS NOT NULL AND {hand_col} BETWEEN {hand_range[0]} AND {hand_range[1]}"
#
#     sql = f"""
#     SELECT
#         CAST(x AS DOUBLE) AS x,
#         CAST(y AS DOUBLE) AS y,
#         orthometric_height,
#         h_{dem}        AS h_dem,
#         delta_{dem}    AS delta,
#         time
#     FROM read_parquet('{TRACKS_PARQUET}')
#     WHERE track={track} AND rgt={rgt} AND spot={spot}
#       AND DATE(time) = DATE '{date}'
#       AND atl03_cnf = 4 AND atl08_class = 1
#       AND h_{dem} IS NOT NULL AND delta_{dem} IS NOT NULL
#       {hand_sql}
#     ORDER BY x
#     """
#     try:
#         return duckdb.query(sql).to_df()
#     except Exception as e:
#         print("DuckDB tracks query failed:", e)
#         import pandas as pd
#         return pd.DataFrame()
#
# def _add_distance(df):
#     if df is None or df.empty:
#         return df
#     geod = Geod(ellps="WGS84")
#     d = np.zeros(len(df))
#     if len(df) > 1:
#         _, _, dp = geod.inv(df["x"].to_numpy()[:-1], df["y"].to_numpy()[:-1],
#                             df["x"].to_numpy()[1:],  df["y"].to_numpy()[1:])
#         d[1:] = dp
#     df = df.copy()
#     df["distance_m"] = np.cumsum(d)
#     return df
#
# def _deck_spec_from_tracks(df, basemap_style):
#     layers = []
#     # 1. КОНТУР БАСЕЙНУ
#     if basin_json:
#         layers.append({
#             "@@type": "GeoJsonLayer", "id": "basin-outline",
#             "data": basin_json, "stroked": True, "filled": False,
#             "getLineColor": [0, 102, 255, 220],
#             "getFillColor": [0, 0, 0, 0],
#             "getLineWidth": 2.5, "lineWidthUnits": "pixels",
#             "lineWidthMinPixels": 2, "parameters": {"depthTest": False}
#         })
#
#     # 2. ТОЧКИ ТРЕКУ ТА ЛІНІЯ (головна мета цього запиту)
#     if df is not None and not df.empty:
#         lon, lat = df["x"].to_numpy(), df["y"].to_numpy()
#         delta     = df["delta"].to_numpy()
#         # САМПЛІНГ, щоб не «вбивати» фронт
#         step = max(1, len(df)//2000)  # до ~2000 маркерів
#         pts = [{"position": [float(lon[i]), float(lat[i])],
#                 "color": _color_rd_bu(float(delta[i]))}
#                for i in range(0, len(df), step)]
#         path = [[float(x), float(y)] for x, y in zip(lon, lat)]
#
#         layers += [
#             {   # ScatterplotLayer (точки)
#                 "@@type": "ScatterplotLayer", "id": "track-points",
#                 "data": pts, "pickable": True,
#                 "parameters": {"depthTest": False},
#                 "radiusUnits": "pixels",
#                 "getRadius": 3, "radiusMinPixels": 2, "radiusMaxPixels": 8,
#                 "getPosition": "@@=d.position",
#                 "getFillColor": "@@=d.color"
#             },
#             {
#                 "@@type": "PathLayer", "id": "track-path",
#                 "data": [{"path": path}],
#                 "getPath": "@@=d.path", "widthUnits": "pixels", "getWidth": 2,
#                 "getColor": [255, 200, 0, 220], "parameters": {"depthTest": False}
#             }
#         ]
#
#     return json.dumps({
#         "mapStyle": basemap_style,
#         "controller": True,
#         "initialViewState": {
#             "bounds": list(basin_bounds),
#             "pitch": 0, "bearing": 0, "minZoom": 7, "maxZoom": 13
#         },
#         "layers": layers
#     })
#
# # --- Track/RGT/Spot Dropdown (КОЛБЕКИ ВНЕСЕНІ БЕЗ ЗМІН) ---
#
# @app.callback(
#     Output("track_rgt_spot_dropdown", "options"),
#     Output("track_rgt_spot_dropdown", "value"),
#     Input("year_dropdown", "value"),
#     State("selected_profile", "data"),
# )
# def update_tracks_dropdown(year, selected_profile):
#     options = db.get_track_dropdown_options(year)
#     value = options[0]["value"] if options else None
#     if selected_profile and selected_profile.get("track") in [o["value"] for o in options]:
#         value = selected_profile["track"]
#     return options, value
#
# @app.callback(
#     Output("date_dropdown", "options"),
#     Output("date_dropdown", "value"),
#     Input("track_rgt_spot_dropdown", "value"),
#     State("selected_profile", "data"),
# )
# def update_dates_dropdown(track_rgt_spot, selected_profile):
#     if not track_rgt_spot:
#         return [], None
#     track, rgt, spot = map(float, track_rgt_spot.split("_"))
#     options = db.get_date_dropdown_options(track, rgt, spot)
#     value = options[0]["value"] if options else None
#     if selected_profile and selected_profile.get("date") in [o["value"] for o in options]:
#         value = selected_profile["date"]
#     return options, value
#
#
# # --- STORE: (КОЛБЕК ВНЕСЕНИЙ БЕЗ ЗМІН) ---
# @callback(
#     Output("selected_profile", "data"),
#     Output("profile_history", "data"),
#     Input("year_dropdown", "value"),
#     Input("track_rgt_spot_dropdown", "value"),
#     Input("date_dropdown", "value"),
#     Input("dem_dropdown", "value"),                      # ◀︎ NEW
#     State("selected_profile", "data"),
#     State("profile_history", "data"),
#     prevent_initial_call=True
# )
# def sync_profile_to_store(year, track, date, dem_value, prev_profile, history):
#     dem = dem_value or (prev_profile or {}).get("dem") or DEFAULT_DEM
#     profile = {"year": year, "track": track, "date": date, "dem": dem}
#     history = (history or []) + ([] if prev_profile == profile else [profile])
#     return profile, history
#
#
# def add_distance_m(df, lon_col="x", lat_col="y"):
#     if df is None or df.empty: return df
#     geod = Geod(ellps="WGS84")
#     lons, lats = df[lon_col].to_numpy(), df[lat_col].to_numpy()
#     d = np.zeros(len(df))
#     if len(df) > 1:
#         _, _, dp = geod.inv(lons[:-1], lats[:-1], lons[1:], lats[1:])
#         d[1:] = dp
#     out = df.copy(); out["distance_m"] = np.cumsum(d)
#     return out
#
# # --- UPDATE PROFILE (КОЛБЕК ВНЕСЕНИЙ БЕЗ ЗМІН) ---
# @callback(
#     Output("track_profile_graph", "figure"),
#     Output("dem_stats", "children"),
#     Input("track_rgt_spot_dropdown", "value"),
#     Input("date_dropdown", "value"),
#     Input("hand_slider", "value"),
#     Input("hand_toggle", "value"),
#     Input("interp_method", "value"),
#     Input("kalman_q", "value"),
#     Input("kalman_r", "value"),
#     Input("dem_dropdown", "value"),  # ◀︎ NEW
#     Input("selected_profile", "data"),                   # ◀︎ NEW (було State)
# )
# def update_profile(track_rgt_spot, date,
#                    hand_range, hand_toggle,
#                    interp_method, kalman_q, kalman_r,
#                    dem_value, selected_profile):
#     if not track_rgt_spot or not date:
#         return empty_dark_figure(text="Виберіть трек і дату"), "No error stats"
#
#     dem = dem_value or (selected_profile or {}).get("dem") or DEFAULT_DEM
#     try:
#         track, rgt, spot = map(float, track_rgt_spot.split("_"))
#     except Exception:
#         return empty_dark_figure(text="Некоректний формат треку."), "No error stats"
#
#     use_hand = isinstance(hand_toggle, (list, tuple, set)) and "on" in hand_toggle
#     hand_q = hand_range if (use_hand and hand_range and len(hand_range) == 2
#                             and all(isinstance(x, (int, float)) for x in hand_range)) else None
#
#     df_hand = db.get_profile(track, rgt, spot, dem, date, hand_q)
#     df_all  = db.get_profile(track, rgt, spot, dem, date, None)
#     if df_hand is not None and not df_hand.empty and "distance_m" not in df_hand:
#         df_hand = add_distance_m(df_hand)
#     if df_all is not None and not df_all.empty and "distance_m" not in df_all:
#         df_all = add_distance_m(df_all)
#     if (df_all is None or df_all.empty or
#         f"h_{dem}" not in df_all or df_all[f"h_{dem}"].dropna().empty):
#         return empty_dark_figure(text="Немає даних для побудови профілю."), "No error stats"
#     interpolated_df = None
#     if interp_method and interp_method not in ["none", "raw", "", None]:
#         df_ice = df_all[["distance_m","orthometric_height"]].dropna().sort_values("distance_m") \
#                 if "distance_m" in df_all else pd.DataFrame()
#         if not df_ice.empty:
#             if interp_method == "linear":
#                 grid = np.linspace(df_ice["distance_m"].min(), df_ice["distance_m"].max(), 300)
#                 interpolated_df = interpolate_linear(df_ice, grid=grid)
#             elif interp_method == "kalman":
#                 transition_cov = 10 ** (kalman_q if kalman_q is not None else -1)
#                 observation_cov = kalman_r if kalman_r is not None else 0.6
#                 smooth_df = kalman_smooth(df_ice,
#                                           transition_covariance=transition_cov,
#                                           observation_covariance=observation_cov)
#                 interpolated_df = smooth_df[["distance_m","kalman_smooth"]].rename(
#                     columns={"kalman_smooth":"orthometric_height"}
#                 )
#     fig = build_profile_figure_with_hand(
#         df_all=df_all, df_hand=df_hand,
#         dem_key=dem, use_hand=use_hand,
#         interpolated_df=interpolated_df, interp_method=interp_method
#     )
#     stats = db.get_dem_stats(df_all, dem)
#     stats_text = (f"Mean error: {stats['mean']:.2f} м, Min: {stats['min']:.2f} м, "
#                   f"Max: {stats['max']:.2f} м, Points: {stats['count']}"
#                   if stats else "No error stats")
#     return fig, stats_text
#
# # --- CALLBACK: ОНОВЛЕННЯ MAP (ПЕРЕВІРКА) ---
# # Цей колбек бере дані треку, форматує їх через _deck_spec_from_tracks (який додає шари точок і ліній)
# # і оновлює 'spec' компонента dash_deckgl.
# @callback(
#     Output("deck-track", "spec"),
#     Input("selected_profile", "data"),
#     Input("hand_slider", "value"),
#     Input("hand_toggle", "value"),
#     Input("basemap_style", "value"),
# )
# def update_track_map(selected_profile, hand_range, hand_toggle, basemap_style):
#     if not selected_profile or not all(selected_profile.values()):
#         return _deck_spec_from_tracks(None, basemap_style)
#     try:
#         track, rgt, spot = map(float, selected_profile["track"].split("_"))
#         dem  = selected_profile.get("dem")
#         date = selected_profile.get("date")
#     except Exception:
#         return _deck_spec_from_tracks(None, basemap_style)
#     use_hand = isinstance(hand_toggle, (list, tuple, set)) and "on" in hand_toggle
#     hand_q = hand_range if (use_hand and hand_range and len(hand_range) == 2) else None
#     df = _query_tracks(track, rgt, spot, date, dem, hand_q)
#     df = _add_distance(df)
#     return _deck_spec_from_tracks(df, basemap_style)
