# pages/tracks_filters.py

from dash import html, dcc
from utils.style import dark_dropdown_style

YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
DEM_LIST = [
    "alos_dem", "aster_dem", "copernicus_dem", "fab_dem",
    "nasa_dem", "srtm_dem", "tan_dem"
]

filters_layout = html.Div([
    dcc.Dropdown(
        id="year_dropdown",
        options=[{"label": str(y), "value": y} for y in YEARS],
        value=YEARS[-1], clearable=False,
        style={**dark_dropdown_style, "width": "100px", "display": "inline-block"}
    ),
    dcc.Dropdown(
        id="dem_dropdown",
        options=[{"label": d.upper(), "value": d} for d in DEM_LIST],
        value=DEM_LIST[0], clearable=False,
        style={**dark_dropdown_style, "width": "130px", "display": "inline-block", "marginLeft": "8px"}
    ),
    dcc.Dropdown(
        id="track_rgt_spot_dropdown",
        clearable=False,
        style={**dark_dropdown_style, "width": "350px", "display": "inline-block", "marginLeft": "8px"}
    ),
    dcc.Dropdown(
        id="date_dropdown",
        clearable=False,
        style={**dark_dropdown_style, "width": "160px", "display": "inline-block", "marginLeft": "8px"}
    ),
    html.Div([
        html.Label("HAND (Height Above Nearest Drainage), м:", style={"color": "#EEEEEE"}),
        dcc.Checklist(
            id="hand_filter_toggle",
            options=[{"label": "Фільтрувати по HAND (floodplain)", "value": "on"}],
            value=["on"],
            style={"margin": "8px 0 4px 0", "color": "#EEEEEE"}
        ),
        dcc.RangeSlider(
            id="hand_slider", min=0, max=20, step=1, value=[0, 5],
            marks={i: str(i) for i in range(0, 21, 5)},
            tooltip={"placement": "bottom", "always_visible": True}
        ),
    ], style={"width": "350px", "marginTop": "8px"}),
], style={"marginBottom": "12px", "width": "100%"})
