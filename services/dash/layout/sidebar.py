from dash import html, dcc
from utils.style import sidebar_style, label_style, dropdown_style, button_style

dem_list = [
    "alos_dem", "aster_dem", "copernicus_dem",
    "fab_dem", "nasa_dem", "srtm_dem", "tan_dem"
]

sidebar = html.Div([
    html.H4("🔧 Фільтри", style={"color": "#EEEEEE"}),

    dcc.Dropdown(
        id="dem_select",
        options=[{"label": (d.upper()).replace('_', " "), "value": d} for d in dem_list],
        value=dem_list[0],
        style=dropdown_style
    ),

    html.Label("LULC:", style=label_style),
    dcc.Dropdown(id="lulc_select", multi=True, options=[], style=dropdown_style),

    html.Label("Геоморфони:", style=label_style),
    dcc.Dropdown(id="landform_select", multi=True, options=[], style=dropdown_style),

    html.Label("Похил:", style=label_style),
    dcc.RangeSlider(
        id="slope_slider", min=0, max=60, step=1,
        marks={i: str(i) for i in range(0, 61, 10)},
        value=[0, 60]
    ),

    html.Label("HAND:", style=label_style),
    dcc.Checklist(
        id="hand_toggle",
        options=[{"label": "Фільтрувати", "value": "on"}],
        value=[], style={"color": "#EEE"}
    ),
    dcc.RangeSlider(
        id="hand_slider", min=0, max=21, step=1,
        marks={i: str(i) for i in range(0, 21, 5)},
        value=[0, 5]
    ),

    html.Button("🔄 Оновити", id="apply_filters_btn", n_clicks=0, style=button_style)
], style=sidebar_style)
