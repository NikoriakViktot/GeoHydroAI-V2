#pages/index.py
import dash
from dash import html, dcc
from layout.tabs_content import content
from callbacks.main_callbacks import *
from layout.sidebar import sidebar

from callbacks.profile_callback import *
from callbacks.best_model_callback import update_best_dem_tab

dash.register_page(
    __name__,
    path="/",
    name="ДАШБОРД",
    title="GeoHydroAI | Карта ICESat-2",
    order=0
)



layout = html.Div(
    id="layout",
    children=[
        html.Button("☰", id="burger", n_clicks=0, className="hamburger"),
        html.Div(id="sidebar-wrap", children=[
            html.Button("⮜", id="collapse", n_clicks=0, className="collapse-btn"),
            sidebar
        ]),
        html.Div(id="content", children=[content]),
        html.Div(id="sidebar-backdrop", n_clicks=0)
    ]
)