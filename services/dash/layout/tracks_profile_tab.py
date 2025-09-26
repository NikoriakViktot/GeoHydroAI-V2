# pages/track_profile_layout.py
import os
import json
import logging
import geopandas as gpd
import dash_deckgl
from dash import html, dcc

from utils.style import dark_card_style, dropdown_style, empty_dark_figure
from registry import get_df

logger = logging.getLogger(__name__)

MAPBOX_ACCESS_TOKEN = os.getenv("MAPBOX_ACCESS_TOKEN", "").strip()
YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

# ----- basin (geojson + bounds)
try:
    basin_gdf: gpd.GeoDataFrame = get_df("basin").to_crs("EPSG:4326")
    basin_json = json.loads(basin_gdf.to_json())
    minx, miny, maxx, maxy = basin_gdf.total_bounds  # west, south, east, north
    basin_bounds = [float(minx), float(miny), float(maxx), float(maxy)]
    logger.info("Basin loaded, CRS=%s, rows=%d, bounds=%s", basin_gdf.crs, len(basin_gdf), basin_bounds)
except Exception as e:
    logger.exception("Failed to load basin: %s", e)
    basin_json = None
    basin_bounds = [24.7, 47.5, 25.5, 48.2]  # запасне вікно

def basin_layer(geojson: dict) -> dict:
    return {
        "@@type": "GeoJsonLayer",
        "id": "basin-outline",
        "data": geojson,
        "stroked": True,
        "filled": False,
        "getLineColor": [0, 102, 255, 220],
        "getFillColor": [0, 0, 0, 0],
        "getLineWidth": 2.5,
        "lineWidthUnits": "pixels",
        "lineWidthMinPixels": 2,
        "parameters": {"depthTest": False}
    }

def initial_view_for_basin(bounds):
    # deck.gl вміє фітитись по bbox через initialViewState.bounds
    return {
        "bounds": bounds,  # [W, S, E, N]
        "maxZoom": 12,
        "minZoom": 7,
        "pitch": 0,
        "bearing": 0
    }

# ----- основний layout вкладки (карта праворуч, менша)
profile_tab_layout = html.Div(
    [
        html.H4("ICESat-2 Track Profiles", style={"color": "#EEEEEE", "margin": "4px 0 10px 4px"}),

        # --- Панель керування (1 рядок)
        html.Div(
            [
                dcc.Dropdown(
                    id="year_dropdown",
                    options=[{"label": str(y), "value": y} for y in YEARS],
                    value=YEARS[-1],
                    clearable=False,
                    style={**dropdown_style, "width": "90px"},
                ),
                dcc.Dropdown(
                    id="track_rgt_spot_dropdown",
                    options=[],
                    style={**dropdown_style, "width": "240px", "marginLeft": "8px"},
                ),
                dcc.Dropdown(
                    id="date_dropdown",
                    options=[],
                    style={**dropdown_style, "width": "140px", "marginLeft": "8px"},
                ),
                dcc.Dropdown(
                    id="interp_method",
                    options=[
                        {"label": "No interpolation", "value": "none"},
                        {"label": "Linear interpolation", "value": "linear"},
                        {"label": "Kalman filter", "value": "kalman"},
                    ],
                    value="none",
                    clearable=False,
                    style={**dropdown_style, "width": "190px", "marginLeft": "8px"},
                ),
                dcc.Dropdown(
                    id="basemap_style",
                    options=[
                        {"label": "Mapbox Outdoors", "value": "mapbox://styles/mapbox/outdoors-v12"},
                        {"label": "Mapbox Satellite", "value": "mapbox://styles/mapbox/satellite-v9"},
                    ],
                    value="mapbox://styles/mapbox/outdoors-v12",
                    clearable=False,
                    style={**dropdown_style, "width": "220px", "marginLeft": "8px"},
                ),
            ],
            style={"display": "flex", "gap": "10px", "marginBottom": "10px"},
        ),

        # --- Слайдери Kalman
        html.Div(
            [
                html.Label(
                    [
                        "Kalman Q (Process noise)",
                        html.Span(" — Lower = smoother; higher = sensitive.", style={"fontSize": "12px", "marginLeft": "8px", "color": "#AAA"}),
                    ],
                    style={"color": "#EEE"},
                ),
                dcc.Slider(id="kalman_q", min=-2, max=0, step=0.1, value=-1,
                           marks={i: f"1e{i}" for i in range(-6, 0)},
                           tooltip={"placement": "bottom"}, included=False),
            ],
            style={"marginBottom": "10px", "marginLeft": "8px"},
        ),
        html.Div(
            [
                html.Label(
                    [
                        "Kalman R (Observation noise)",
                        html.Span(" — Higher = less sensitive to outliers.", style={"fontSize": "12px", "marginLeft": "8px", "color": "#AAA"}),
                    ],
                    style={"color": "#EEE"},
                ),
                dcc.Slider(id="kalman_r", min=0, max=2, step=0.1, value=0.6,
                           marks={i: str(i) for i in range(0, 3)},
                           tooltip={"placement": "bottom"}, included=False),
            ],
            style={"marginBottom": "6px", "marginLeft": "8px"},
        ),

        # --- Дві колонки: зліва графік, справа компактна карта
        html.Div(
            [
                # ЛІВА КОЛОНКА (графік)
                html.Div(
                    dcc.Loading(
                        id="track_profile_loading",
                        type="circle",
                        color="#1c8cff",
                        children=[
                            dcc.Graph(
                                id="track_profile_graph",
                                figure=empty_dark_figure(),
                                style={
                                    "height": "540px",
                                    "width": "100%",
                                    "minWidth": "580px",
                                    "marginBottom": "8px",
                                    "backgroundColor": "#181818",
                                },
                            )
                        ],
                    ),
                    style={"minWidth": "580px"}
                ),

                # ПРАВА КОЛОНКА (менша карта + статистика)
                html.Div(
                    [
                        dash_deckgl.DashDeckgl(
                            id="deck-track",
                            # одразу малюємо контур басейну і фітимось по bounds
                            spec=json.dumps({
                                "mapStyle": "mapbox://styles/mapbox/outdoors-v12",
                                "initialViewState": initial_view_for_basin(basin_bounds),
                                "controller": {"minZoom": 7, "maxZoom": 12},
                                "layers": ([basin_layer(basin_json)] if basin_json else [])
                            }),
                            height=360,
                            mapbox_key=MAPBOX_ACCESS_TOKEN,
                            cursor_position="bottom-right",
                            events=["hover", "click"],
                            description={"top-right": "<div id='track-legend'></div>"},
                        ),
                        html.Div(
                            id="dem_stats",
                            style={
                                **dark_card_style,
                                "marginTop": "12px",
                                "fontSize": "14px",
                                "display": "inline-flex",
                                "width": "100%",
                                "maxWidth": "100%",
                            },
                        ),
                    ],
                    style={
                        "width": "380px",         # компактна карта праворуч
                        "minWidth": "340px",
                        "paddingLeft": "10px"
                    },
                ),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "1fr 380px",
                "gap": "6px",
                "alignItems": "start"
            },
        ),
    ],
    style={
        "backgroundColor": "#181818",
        "color": "#EEEEEE",
        "minHeight": "480px",
        "padding": "12px",
    },
)

# щоб колбеки могли орієнтуватись на ці дані
__all__ = ["profile_tab_layout", "basin_json", "basin_bounds"]
