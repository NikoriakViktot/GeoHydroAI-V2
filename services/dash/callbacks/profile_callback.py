import plotly.graph_objs as go
import dash_leaflet as dl
import numpy as np
from pyproj import Geod
import geopandas as gpd
import duckdb
import os

from utils.plot_track import build_profile_figure_with_hand
from utils.style import empty_dark_figure
from dash import callback, Output, Input, State, no_update, exceptions
import dash_leaflet as dl
import dash
import pandas as pd
import json
import logging
from layout.tracks_profile_tab import basin_json

from src.interpolation_track import (
        kalman_smooth,
    interpolate_linear,
    )
from registry import get_db, get_df

logger = logging.getLogger(__name__)  # ✅ тепер logger існує

app = dash.get_app()
db = get_db("nmad")
DEFAULT_DEM = os.getenv("DEFAULT_TRACK_DEM", "alos_dem")
DEM_LIST = [
    "alos_dem", "aster_dem", "copernicus_dem", "fab_dem",
    "nasa_dem", "srtm_dem", "tan_dem"
]
YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
hand_column_map = {dem: f"{dem}_2000" for dem in DEM_LIST}

def _color_rd_bu(delta: float, vmax: float = 20.0) -> list[int]:
    if delta is None or not np.isfinite(delta):
        return [200, 200, 200, 180]
    x = float(np.clip(delta / vmax, -1.0, 1.0))
    if x < 0:
        t = abs(x); r, g, b = int(255*(1-t)), int(255*(1-t)), 255
    else:
        t = x; r, g, b = 255, int(255*(1-t)), int(255*(1-t))
    return [r, g, b, 200]

def build_track_spec(df, basemap_style, basin_geojson=basin_json) -> str:
    layers = []
    # басейн
    if basin_geojson:
        layers.append({
            "@@type": "GeoJsonLayer", "id": "basin-outline",
            "data": basin_geojson, "stroked": True, "filled": False,
            "getLineColor": [0, 102, 255, 220],
            "getFillColor": [0, 0, 0, 0],
            "getLineWidth": 2.5, "lineWidthUnits": "pixels", "lineWidthMinPixels": 2,
            "parameters": {"depthTest": False}
        })

    # трек
    view = {"longitude": 25.03, "latitude": 47.8, "zoom": 9}
    if df is not None and not df.empty:
        df = df.sort_values("distance_m")
        lon, lat = df["x"].to_numpy(), df["y"].to_numpy()
        delta_col = next((c for c in df.columns if c.startswith("delta_")), None)
        deltas = df[delta_col].to_numpy() if delta_col else np.full(len(df), np.nan)

        pts = [{"position": [float(lon[i]), float(lat[i])],
                "color": _color_rd_bu(float(deltas[i]))}
               for i in range(0, len(df), 10)]
        path_coords = [[float(x), float(y)] for x, y in zip(lon, lat)]

        layers += [
            {"@@type": "ScatterplotLayer", "id": "track-points",
             "data": pts, "pickable": True, "getPosition": "@@=d.position",
             "getFillColor": "@@=d.color", "radiusUnits": "meters", "getRadius": 22},
            {"@@type": "PathLayer", "id": "track-path",
             "data": [{"path": path_coords}],
             "getPath": "@@=d.path", "getWidth": 2, "widthUnits": "pixels",
             "getColor": [255, 200, 0, 220]}
        ]
        view = {"longitude": float(np.nanmean(lon)), "latitude": float(np.nanmean(lat)), "zoom": 10}

    return json.dumps({"mapStyle": basemap_style, "initialViewState": view, "layers": layers})



# --- Track/RGT/Spot Dropdown
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


# --- STORE: єдиний callback для синхронізації state/history
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
    # Дем беремо зі стора, а якщо його там нема — дефолт
    dem = (prev_profile or {}).get("dem", DEFAULT_DEM)
    profile = {"year": year, "track": track, "date": date, "dem": dem}

    history = history or []
    if not prev_profile or prev_profile != profile:
        history.append(profile)

    return profile, history

def add_distance_m(df, lon_col="x", lat_col="y"):
    if df is None or df.empty: return df
    geod = Geod(ellps="WGS84")
    lons, lats = df[lon_col].to_numpy(), df[lat_col].to_numpy()
    d = np.zeros(len(df))
    if len(df) > 1:
        _, _, dp = geod.inv(lons[:-1], lats[:-1], lons[1:], lats[1:])
        d[1:] = dp
    out = df.copy(); out["distance_m"] = np.cumsum(d)
    return out

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
    df_all  = db.get_profile(track, rgt, spot, dem, date, None)

    if df_hand is not None and not df_hand.empty and "distance_m" not in df_hand:
        df_hand = add_distance_m(df_hand)
    if df_all is not None and not df_all.empty and "distance_m" not in df_all:
        df_all = add_distance_m(df_all)

    if (df_all is None or df_all.empty or
        f"h_{dem}" not in df_all or df_all[f"h_{dem}"].dropna().empty):
        return empty_dark_figure(text="Немає даних для побудови профілю."), "No error stats"

    interpolated_df = None
    if interp_method and interp_method not in ["none", "raw", "", None]:
        df_ice = df_all[["distance_m","orthometric_height"]].dropna().sort_values("distance_m") \
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
                interpolated_df = smooth_df[["distance_m","kalman_smooth"]].rename(
                    columns={"kalman_smooth":"orthometric_height"}
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


@app.callback(
    Output("point_group", "children"),
    Input("selected_profile", "data"),
    Input("hand_slider", "value"),
    Input("hand_toggle", "value"),
)
def update_map_points(selected_profile, hand_range, hand_toggle):
    if not selected_profile or not all(selected_profile.values()):
        return []
    track_str = selected_profile["track"]
    dem = selected_profile["dem"]
    date = selected_profile["date"]
    try:
        track, rgt, spot = map(float, track_str.split("_"))
    except Exception:
        return []
    use_hand = "on" in hand_toggle
    hand_range_for_query = (
        hand_range if (use_hand and hand_range and len(hand_range) == 2 and all(isinstance(x, (int, float)) for x in hand_range))
        else None
    )
    df = db.get_profile(track, rgt, spot, dem, date, hand_range_for_query)
    if df is None or df.empty:
        return []
    lon_col = "x"
    lat_col = "y"
    delta_col = f"delta_{dem}"
    ortho_col = "orthometric_height"

    SAMPLE_STEP = 10  # ← Змінюй, як треба
    df_sampled = df.iloc[::SAMPLE_STEP].copy()

    def tooltip_text(row):
        delta_val = row[delta_col] if pd.notna(row[delta_col]) else "NaN"
        ortho_val = row[ortho_col] if ortho_col in row and pd.notna(row[ortho_col]) else "NaN"
        return f"ICESat-2 (Ortho): {ortho_val:.2f} м."

    markers = [
        dl.CircleMarker(
            center=[row[lat_col], row[lon_col]],
            radius=3,
            color="blue",
            fillColor="blue",
            fillOpacity=0.9,
            children=[dl.Tooltip(tooltip_text(row))],
        )
        for _, row in df_sampled.iterrows() if pd.notna(row[delta_col]) and pd.notna(row[ortho_col])
    ]
    # Виділяємо мін/макс — вони можуть бути не у sample, тож їх краще брати з оригінального df:
    if not df[delta_col].dropna().empty:
        min_idx = df[delta_col].idxmin()
        max_idx = df[delta_col].idxmax()
        row_min = df.loc[min_idx]
        row_max = df.loc[max_idx]
        markers.append(
            dl.CircleMarker(
                center=[row_min[lat_col], row_min[lon_col]],
                radius=7, color="lime", fillColor="lime", fillOpacity=1,
                children=[dl.Tooltip(f"Min ΔDEM: {row_min[delta_col]:.2f} м. ICESat-2: {row_min[ortho_col]:.2f} м.")]
            )
        )
        if min_idx != max_idx:
            markers.append(
                dl.CircleMarker(
                    center=[row_max[lat_col], row_max[lon_col]],
                    radius=7, color="red", fillColor="red", fillOpacity=1,
                    children=[dl.Tooltip(f"Max ΔDEM: {row_max[delta_col]:.2f} м. ICESat-2: {row_max[ortho_col]:.2f} м.")]
                )
            )
    return markers


def _color_rd_bu(delta: float, vmax: float = 20.0) -> list[int]:
    if delta is None or not np.isfinite(delta):
        return [200, 200, 200, 180]
    x = float(np.clip(delta / vmax, -1.0, 1.0))
    if x < 0:
        t = abs(x); r, g, b = int(255*(1-t)), int(255*(1-t)), 255
    else:
        t = x; r, g, b = 255, int(255*(1-t)), int(255*(1-t))
    return [r, g, b, 200]


@callback(
    Output("deck-track", "spec"),
    Input("selected_profile", "data"),
    Input("hand_slider", "value"),
    Input("hand_toggle", "value"),
    Input("basemap_style", "value"),
    prevent_initial_call=True
)
def update_track_map(selected_profile, hand_range, hand_toggle, basemap_style):
    if not selected_profile or not all(selected_profile.values()):
        # все одно повертаємо spec з басейном
        return build_track_spec(None, basemap_style)

    try:
        track, rgt, spot = map(float, selected_profile["track"].split("_"))
    except Exception:
        return build_track_spec(None, basemap_style)

    dem  = selected_profile.get("dem") or os.getenv("DEFAULT_TRACK_DEM", "alos_dem")
    date = selected_profile["date"]

    use_hand = isinstance(hand_toggle, (list, tuple, set)) and "on" in hand_toggle
    hand_q = hand_range if (use_hand and hand_range and len(hand_range) == 2) else None

    df = db.get_profile(track, rgt, spot, dem, date, hand_q)
    if df is not None and "distance_m" not in df:
        geod = Geod(ellps="WGS84")
        d = np.zeros(len(df))
        if len(df) > 1:
            _, _, dp = geod.inv(df["x"].to_numpy()[:-1], df["y"].to_numpy()[:-1],
                                df["x"].to_numpy()[1:],  df["y"].to_numpy()[1:])
            d[1:] = dp
        df = df.copy(); df["distance_m"] = np.cumsum(d)

    return build_track_spec(df, basemap_style)