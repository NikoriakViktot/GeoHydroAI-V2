#layout/tabs_content.py

from dash import html, dcc
from utils.style import tab_style, selected_tab_style


content = html.Div([
    # Stores для стейту профілю, історії та CDF
    dcc.Store(id="selected_profile", storage_type="session"),
    dcc.Store(id="profile_history", storage_type="session"),
    dcc.Store(id="cdf-store", storage_type="session"),

    # Вкладки
    dcc.Tabs(
        id="tabs",
        value="tab-1",
        className="custom-tabs",
        children=[
            dcc.Tab(label="📊 DEM Comparison", value="tab-1", style=tab_style, selected_style=selected_tab_style),
            dcc.Tab(label="🗺️ ICESat-2 Tracks Map", value="tab-3", style=tab_style, selected_style=selected_tab_style),
            dcc.Tab(label="📈 Track Profile", value="tab-2", style=tab_style, selected_style=selected_tab_style),
            dcc.Tab(label="🏆 Best DEM", value="tab-4", style=tab_style, selected_style=selected_tab_style),
            dcc.Tab(label="CDF Accumulation", value="tab-5", style=tab_style, selected_style=selected_tab_style),
        ],
        style={"width": "100%"},
    ),

    # ТІЛЬКИ ОДИН головний контент Div
    html.Div(id="tab-content"),
], style={
    "padding": "0 0 24px 0",
    "backgroundColor": "#181818",
    "color": "#EEEEEE",
    "minHeight": "100vh",
})
