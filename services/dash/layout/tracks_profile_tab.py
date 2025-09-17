from dash import html, dcc
from utils.style import dark_card_style, dropdown_style, empty_dark_figure

YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

profile_tab_layout = html.Div([
    html.H4("Профіль ICESat-2 треку", style={"color": "#EEEEEE"}),

    # ВСІ ДРОПДАУНИ В ОДНОМУ РЯДКУ
    html.Div([
        dcc.Dropdown(
            id="year_dropdown",
            options=[{"label": str(y), "value": y} for y in YEARS],
            value=YEARS[-1], clearable=False,
            style={**dropdown_style, "width": "90px"}
        ),
        dcc.Dropdown(
            id="track_rgt_spot_dropdown",
            options=[],
            style={**dropdown_style, "width": "240px", "marginLeft": "8px"}
        ),
        dcc.Dropdown(
            id="date_dropdown",
            options=[],
            style={**dropdown_style, "width": "140px", "marginLeft": "8px"}
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
            style={**dropdown_style, "width": "190px", "marginLeft": "8px"}
        ),
    ], style={"display": "flex", "gap": "10px", "marginBottom": "10px"}),

    # --- Kalman parameters with explanation ---
    html.Div([
        html.Label([
            "Kalman Q (Process noise)",
            html.Span(
                " — Lower values = more smoothing. Higher = more sensitive to changes.",
                style={"fontSize": "12px", "marginLeft": "8px", "color": "#AAA"}
            )
        ], style={"color": "#EEE"}),
        dcc.Slider(
            id="kalman_q",
            min=-2, max=0, step=0.1, value=-1,
            marks={i: f"1e{i}" for i in range(-6, 0)},
            tooltip={"placement": "bottom"},
            included=False,
        ),
    ], style={"marginBottom": "10px", "marginLeft": "8px"}),

    html.Div([
        html.Label([
            "Kalman R (Observation noise)",
            html.Span(
                " — Higher values = less sensitive to outliers.",
                style={"fontSize": "12px", "marginLeft": "8px", "color": "#AAA"}
            )
        ], style={"color": "#EEE"}),
        dcc.Slider(
            id="kalman_r",
            min=0, max=2, step=0.1, value=0.6,
            marks={i: str(i) for i in range(0, 3)},
            tooltip={"placement": "bottom"},
            included=False,
        ),
    ], style={"marginBottom": "16px", "marginLeft": "8px"}),

    # ГРАФІК та СТАТИСТИКА як були
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
                    "minWidth": "650px",
                    "marginBottom": "36px",
                    "backgroundColor": "#181818"
                }
            )
        ]
    ),
    html.Div([
        html.Div(
            id="dem_stats",
            style={
                **dark_card_style,
                "marginTop": "40px",
                "fontSize": "15px",
                "display": "inline-flex",
                "width": "fit-content",
                "maxWidth": "600px",
            }
        )
    ], style={"display": "flex", "justifyContent": "center"}),
], style={
    "backgroundColor": "#181818",
    "color": "#EEEEEE",
    "minHeight": "480px",
    "padding": "18px 12px 32px 12px",
})
