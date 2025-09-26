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

# ---- basin + bounds
try:
    _gdf: gpd.GeoDataFrame = get_df("basin").to_crs("EPSG:4326")
    basin_json = json.loads(_gdf.to_json())
    minx, miny, maxx, maxy = _gdf.total_bounds  # [W,S,E,N]
    basin_bounds = [float(minx), float(miny), float(maxx), float(maxy)]
    logger.info("Basin loaded, CRS=%s, rows=%d", _gdf.crs, len(_gdf))
except Exception as e:
    logger.exception("Failed to load basin: %s", e)
    basin_json = None
    # запасний прямокутник, щоб карта з’явилась
    basin_bounds = [24.5, 47.3, 26.0, 48.2]

def _basin_layer(geojson: dict) -> dict:
    return {
        "@@type": "GeoJsonLayer", "id": "basin-outline",
        "data": geojson,
        "stroked": True, "filled": False,
        "getLineColor": [0, 102, 255, 220],
        "getFillColor": [0, 0, 0, 0],
        "getLineWidth": 2.5,
        "lineWidthUnits": "pixels",
        "lineWidthMinPixels": 2,
        "parameters": {"depthTest": False}
    }

def _initial_spec(map_style: str) -> str:
    return json.dumps({
        "mapStyle": map_style,
        "controller": True,
        "initialViewState": {
            "bounds": basin_bounds,   # фіксація камери по басейну
            "pitch": 0, "bearing": 0, "minZoom": 7, "maxZoom": 13
        },
        "layers": ([ _basin_layer(basin_json) ] if basin_json else [])
    })

profile_tab_layout = html.Div(
    [
        html.H4("ICESat-2 Track Profiles", style={"color": "#EEEEEE"}),

        # ---- верхній ряд з dropdown-ами
        html.Div(
            [
                dcc.Dropdown(id="year_dropdown",
                             options=[{"label": str(y), "value": y} for y in YEARS],
                             value=YEARS[-1], clearable=False,
                             style={**dropdown_style, "width": "90px"}),
                dcc.Dropdown(id="track_rgt_spot_dropdown", options=[],
                             style={**dropdown_style, "width": "240px", "marginLeft": "8px"}),
                dcc.Dropdown(id="date_dropdown", options=[],
                             style={**dropdown_style, "width": "140px", "marginLeft": "8px"}),
                dcc.Dropdown(id="interp_method",
                             options=[
                                 {"label": "No interpolation", "value": "none"},
                                 {"label": "Linear interpolation", "value": "linear"},
                                 {"label": "Kalman filter", "value": "kalman"},
                             ],
                             value="none", clearable=False,
                             style={**dropdown_style, "width": "190px", "marginLeft": "8px"}),
                dcc.Dropdown(id="basemap_style",
                             options=[
                                {"label": "Mapbox Outdoors", "value": "mapbox://styles/mapbox/outdoors-v12"},
                                {"label": "Mapbox Satellite", "value": "mapbox://styles/mapbox/satellite-v9"},
                             ],
                             value="mapbox://styles/mapbox/outdoors-v12", clearable=False,
                             style={**dropdown_style, "width": "220px", "marginLeft": "8px"}),
            ],
            style={"display": "flex", "gap": "10px", "marginBottom": "10px"},
        ),

        # ---- калман слайдери
        html.Div([
            html.Label(["Kalman Q (Process noise)",
                        html.Span(" — Lower = smoother; higher = sensitive.",
                                  style={"fontSize": "12px", "marginLeft": "8px", "color": "#AAA"})],
                       style={"color": "#EEE"}),
            dcc.Slider(id="kalman_q", min=-2, max=0, step=0.1, value=-1,
                       marks={i: f"1e{i}" for i in range(-6, 0)},
                       tooltip={"placement": "bottom"}, included=False),
        ], style={"marginBottom": "10px", "marginLeft": "8px"}),

        html.Div([
            html.Label(["Kalman R (Observation noise)",
                        html.Span(" — Higher = less sensitive to outliers.",
                                  style={"fontSize": "12px", "marginLeft": "8px", "color": "#AAA"})],
                       style={"color": "#EEE"}),
            dcc.Slider(id="kalman_r", min=0, max=2, step=0.1, value=0.6,
                       marks={i: str(i) for i in range(0, 3)},
                       tooltip={"placement": "bottom"}, included=False),
        ], style={"marginBottom": "16px", "marginLeft": "8px"}),

        # ---- ГОЛОВНА СІТКА: зліва графік + статистика, справа карта
        html.Div([
            # ліва колонка
            html.Div([
                dcc.Loading(
                    id="track_profile_loading",
                    type="circle", color="#1c8cff",
                    children=[dcc.Graph(
                        id="track_profile_graph",
                        figure=empty_dark_figure(),
                        style={
                            "height": "400px",
                            "width": "100%",
                            "minWidth": "500px",
                            "backgroundColor": "#181818",
                        },
                    )],
                ),
                html.Div(   # статистика ПІД графіком
                    id="dem_stats",
                    style={**dark_card_style,
                           "marginTop": "14px",
                           "fontSize": "15px",
                           "display": "inline-flex",
                           "width": "fit-content",
                           "maxWidth": "680px"}
                )
            ], style={"display": "flex", "flexDirection": "column"}),

            # права колонка (компактна карта)
            html.Div([
                dash_deckgl.DashDeckgl(
                    id="deck-track",
                    spec=_initial_spec("mapbox://styles/mapbox/outdoors-v12"),
                    height=400,
                    mapbox_key=MAPBOX_ACCESS_TOKEN,
                    cursor_position="bottom-right",
                    events=["hover", "click"],
                    description={"top-right": "<div id='track-legend'></div>"},
                )
            ], style={"minWidth": "360px", "maxWidth": "420px"})
        ], style={
            "display": "grid",
            "gridTemplateColumns": "1fr 400px",
            "gap": "12px",
            "alignItems": "start",
            "marginTop": "4px"
        }),
    ],
    style={"backgroundColor": "#181818", "color": "#EEEEEE",
           "minHeight": "480px", "padding": "18px 12px 32px 12px"},
)

# експортуємо, щоб колбеки могли це використати
__all__ = ["profile_tab_layout", "basin_json", "basin_bounds"]
