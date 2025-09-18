# pages/index.py
import dash
from dash import html
from layout.shell import PageShell
from layout.sidebar import sidebar

dash.register_page(__name__, path="/", name="Dashboard", title="GeoHydroAI | Dashboard", order=0)

body = html.Div(
    [
        html.H3("DEM Comparison"),
        html.Div(id="dashboard-main", children=[
            # сюди колбек(и) цієї сторінки малюватимуть графіки
        ])
    ]
)

layout = PageShell(sidebar, body)
