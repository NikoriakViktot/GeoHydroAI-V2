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


layout = html.Div([
  sidebar,
  content
])
