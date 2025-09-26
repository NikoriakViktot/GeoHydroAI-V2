import os, json, geopandas as gpd
import dash_deckgl
from dash import html, dcc
from utils.style import dark_card_style, dropdown_style, empty_dark_figure
from registry import get_df
import logging
MAPBOX_ACCESS_TOKEN = os.getenv("MAPBOX_ACCESS_TOKEN", "").strip()
logger = logging.getLogger(__name__)  # ✅ тепер logger існує

YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

try:
    basin: gpd.GeoDataFrame = get_df("basin")
    logger.info("Basin loaded, CRS=%s, rows=%d", basin.crs, len(basin))
    basin = basin.to_crs("EPSG:4326")
    basin_json = json.loads(basin.to_json())
except Exception as e:
    logger.exception("Failed to load basin: %s", e)
    basin_json = None


profile_tab_layout = html.Div([
    html.H4("ICESat-2 Track Profiles", style={"color": "#EEEEEE"}),

    html.Div([

        dcc.Dropdown(id="year_dropdown",
                     options=[{"label": str(y), "value": y} for y in YEARS],
                     value=YEARS[-1], clearable=False,
                     style={**dropdown_style, "width": "90px"}),
        dcc.Dropdown(id="track_rgt_spot_dropdown", options=[],
                     style={**dropdown_style, "width": "240px", "marginLeft": "8px"}),
        dcc.Dropdown(id="date_dropdown", options=[],
                     style={**dropdown_style, "width": "140px", "marginLeft": "8px"}),
        dcc.Dropdown(id="interp_method",
                     options=[{"label": "No interpolation", "value": "none"},
                              {"label": "Linear interpolation", "value": "linear"},
                              {"label": "Kalman filter", "value": "kalman"}],
                     value="none", clearable=False,
                     style={**dropdown_style, "width": "190px", "marginLeft": "8px"}),
        # ⬇️ селектор стилю підложки
        dcc.Dropdown(
            id="basemap_style",
            options=[
                {"label": "Mapbox Outdoors", "value": "mapbox://styles/mapbox/outdoors-v12"},
                {"label": "Mapbox Satellite", "value": "mapbox://styles/mapbox/satellite-v9"},
            ],
            value="mapbox://styles/mapbox/outdoors-v12",
            clearable=False,
            style={**dropdown_style, "width": "220px", "marginLeft": "8px"}
        ),
    ], style={"display": "flex", "gap": "10px", "marginBottom": "10px"}),

    # Kalman sliders (як у тебе) ...

    dcc.Loading(
        id="track_profile_loading",
        type="circle",
        color="#1c8cff",
        children=[dcc.Graph(id="track_profile_graph",
                            figure=empty_dark_figure(),
                            style={"height": "540px", "width": "100%", "minWidth": "650px",
                                   "marginBottom": "18px", "backgroundColor": "#181818"})]
    ),

    # ⬇️ НОВЕ: карта deck.gl під графіком
    dash_deckgl.DashDeckgl(
        id="deck-track",
        spec=json.dumps({
            "mapStyle": "mapbox://styles/mapbox/outdoors-v12",
            "initialViewState": {"longitude": 25.03, "latitude": 47.8, "zoom": 9, "pitch": 0, "bearing": 0},
            "layers": []
        }),
        height=420,
        mapbox_key=MAPBOX_ACCESS_TOKEN,
        cursor_position="bottom-right",
        events=["hover", "click"],
        description={"top-right": "<div id='track-legend'></div>"}
    ),

    html.Div([html.Div(id="dem_stats", style={**dark_card_style, "marginTop": "20px",
                                              "fontSize": "15px", "display": "inline-flex",
                                              "width": "fit-content", "maxWidth": "600px"})],
             style={"display": "flex", "justifyContent": "center"}),
], style={"backgroundColor": "#181818", "color": "#EEEEEE",
          "minHeight": "480px", "padding": "18px 12px 32px 12px"})
