# layout/shell.py
from dash import html

def PageShell(sidebar, body):
    return html.Div(
        id="layout",
        children=[
            html.Div(id="sidebar-wrap", children=[sidebar]),
            html.Div(id="content", children=[body]),
            html.Div(id="sidebar-backdrop", n_clicks=0),
        ],
    )
