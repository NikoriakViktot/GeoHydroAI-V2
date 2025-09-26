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
    """Створює шар GeoJson для відображення контуру басейну."""
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
    """
    Повертає початкову специфікацію DeckGL з контуром басейну.
    Використовується для початкового завантаження карти.
    """
    layers = ([ _basin_layer(basin_json) ] if basin_json else [])
    return json.dumps({
        "mapStyle": map_style,
        "controller": True,
        "initialViewState": {
            "bounds": basin_bounds,
            "pitch": 0, "bearing": 0, "minZoom": 7, "maxZoom": 13
        },
        "layers": layers
    })


profile_tab_layout = html.Div(
    [
        # --- БЛОК A: ЗАГОЛОВОК І КЕРУВАННЯ ---
        html.H4("🛰️ ICESat-2 Track Profiler",
                style={"color": "#EEEEEE", "marginBottom": "20px"}),

        # 1. РЯД ДРОПДАУНІВ
        html.Div(
            [
                dcc.Dropdown(id="year_dropdown",
                             options=[{"label": str(y), "value": y} for y in YEARS],
                             value=YEARS[-1], clearable=False,
                             placeholder="Select Year",
                             style={**dropdown_style, "width": "100px"}),
                dcc.Dropdown(id="track_rgt_spot_dropdown", options=[],
                             placeholder="Track / RGT / Spot",
                             style={**dropdown_style, "flexGrow": 2, "minWidth": "200px"}),
                dcc.Dropdown(id="date_dropdown", options=[],
                             placeholder="Observation Date",
                             style={**dropdown_style, "flexGrow": 1, "minWidth": "150px"}),
                dcc.Dropdown(id="interp_method",
                             options=[
                                 {"label": "No Interpolation", "value": "none"},
                                 {"label": "Linear Interpolation", "value": "linear"},
                                 {"label": "Kalman Filter", "value": "kalman"},
                             ],
                             value="none", clearable=False,
                             placeholder="Smoothing Method",
                             style={**dropdown_style, "flexGrow": 1, "minWidth": "160px"}),
                dcc.Dropdown(id="basemap_style",
                             options=[
                                 {"label": "Mapbox Outdoors", "value": "mapbox://styles/mapbox/outdoors-v12"},
                                 {"label": "Mapbox Satellite", "value": "mapbox://styles/mapbox/satellite-v9"},
                             ],
                             value="mapbox://styles/mapbox/outdoors-v12", clearable=False,
                             placeholder="Basemap Style",
                             style={**dropdown_style, "flexGrow": 1, "minWidth": "180px"}),
            ],
            style={"display": "flex", "gap": "10px", "marginBottom": "20px"},
        ),

        # 2. СЛАЙДЕРИ KALMAN
        html.Div([
            html.Div([
                html.Label("Kalman Q (Process Noise)",
                           style={"color": "#EEE", "marginBottom": "5px"}),
                html.Span("— Lower = smoother track profile.",
                          style={"fontSize": "12px", "marginLeft": "10px", "color": "#AAA"}),
                dcc.Slider(id="kalman_q", min=-2, max=0, step=0.1, value=-1,
                           marks={i: f"1e{i}" for i in range(-6, 1, 2)},
                           tooltip={"placement": "bottom", "always_visible": False}, included=False),
            ], style={"flex": "1 1 50%"}),

            html.Div([
                html.Label("Kalman R (Observation Noise)",
                           style={"color": "#EEE", "marginBottom": "5px"}),
                html.Span("— Higher = less sensitive to outliers.",
                          style={"fontSize": "12px", "marginLeft": "10px", "color": "#AAA"}),
                dcc.Slider(id="kalman_r", min=0, max=2, step=0.1, value=0.6,
                           marks={i: str(i) for i in range(0, 3, 1)},
                           tooltip={"placement": "bottom", "always_visible": False}, included=False),
            ], style={"flex": "1 1 50%"}),
        ], style={"display": "flex", "gap": "30px", "marginBottom": "30px", "padding": "0 8px"}),

        # --- БЛОК Б: ВІЗУАЛІЗАЦІЯ (ВЕРТИКАЛЬНИЙ ПОТІК) ---

        # 1. ГРАФІК
        dcc.Loading(
            id="track_profile_loading",
            type="circle", color="#1c8cff",
            children=[dcc.Graph(
                id="track_profile_graph",
                figure=empty_dark_figure(),
                style={
                    "height": "400px",
                    "width": "100%",
                    "backgroundColor": "#181818",
                    "minHeight": "300px",
                },
            )],
        ),

        # 2. СТАТИСТИКА
        html.Div(
            id="dem_stats",
            style={**dark_card_style,
                   "marginTop": "10px",
                   "marginBottom": "20px",
                   "padding": "10px 15px",
                   "fontSize": "16px",
                   "fontWeight": "bold",
                   "borderLeft": "4px solid #1c8cff",
                   "width": "100%"}
        ),

        # 3. КАРТА (ПІД СТАТИСТИКОЮ)
        html.Div([
            html.Label("Track Location and Delta Map",
                       style={"color": "#EEE", "marginBottom": "5px"}),
            dash_deckgl.DashDeckgl(
                id="deck-track",
                spec=_initial_spec("mapbox://styles/mapbox/outdoors-v12"),
                height=450,
                mapbox_key=MAPBOX_ACCESS_TOKEN,
                cursor_position="bottom-right",
                events=["hover", "click"],
                description={"top-right": "<div id='track-legend'></div>"},
            ),
        ], style={"width": "100%", "marginTop": "10px", "minHeight": "450px"}), # Стилі застосовуються тут

    ],
    # Загальні стилі контейнера - АДАПТИВНА ШИРИНА ТА ВИСОТА
    style={"backgroundColor": "#181818", "color": "#EEEEEE",
           "width": "100%",
           "minHeight": "100vh",
           "padding": "24px"},
)

# експортуємо, щоб колбеки могли це використати
__all__ = ["profile_tab_layout", "basin_json", "basin_bounds"]
