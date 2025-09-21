from dash import html, dcc
from utils.style import sidebar_style, label_style, dropdown_style, button_style

# List of available Digital Elevation Models (DEMs)
dem_list = [
    "alos_dem", "aster_dem", "copernicus_dem",
    "fab_dem", "nasa_dem", "srtm_dem", "tan_dem"
]

# Sidebar layout containing filters for DEM analysis
sidebar = html.Div([

    # Section title
    html.H4("🔧 Filters", style={"color": "#EEEEEE"}),

    # DEM selection dropdown
    dcc.Dropdown(
        id="dem_select",
        options=[{"label": (d.upper()).replace('_', " "), "value": d} for d in dem_list],
        value=dem_list[0],
        style=dropdown_style
    ),

    # Land Use / Land Cover (LULC) filter
    html.Label("LULC:", style=label_style),
    dcc.Dropdown(id="lulc_select", multi=True, options=[], style=dropdown_style),

    # Landform (geomorphons) filter
    html.Label("Landforms:", style=label_style),
    dcc.Dropdown(id="landform_select", multi=True, options=[], style=dropdown_style),

    # Slope range filter
    html.Label("Slope:", style=label_style),
    dcc.RangeSlider(
        id="slope_slider", min=0, max=60, step=1,
        marks={i: str(i) for i in range(0, 61, 10)},
        value=[0, 60]
    ),

    # HAND (Height Above Nearest Drainage) filter
    html.Label("HAND:", style=label_style),
    dcc.Checklist(
        id="hand_toggle",
        options=[{"label": "Enable filtering", "value": "on"}],
        value=[], style={"color": "#EEE"}
    ),
    dcc.RangeSlider(
        id="hand_slider", min=0, max=21, step=1,
        marks={i: str(i) for i in range(0, 21, 5)},
        value=[0, 5]
    ),

    # Apply filters button
    html.Button("🔄 Apply Filters", id="apply_filters_btn", n_clicks=0, style=button_style)

], style=sidebar_style)
