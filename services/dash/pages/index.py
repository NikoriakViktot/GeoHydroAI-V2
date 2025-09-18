# pages/index.py
import dash
from dash import html, dcc, Output, Input, State, callback
from layout.sidebar import sidebar

dash.register_page(__name__, path="/", name="Dashboard", title="GeoHydroAI | Dashboard", order=0)

layout = html.Div(
    id="layout",
    children=[
        html.Button("☰", id="burger", n_clicks=0, className="hamburger"),
        html.Div(id="sidebar-wrap", children=[
            html.Button("⮜", id="collapse", n_clicks=0, className="collapse-btn"),
            sidebar,
        ]),
        html.Div(
            id="content",
            children=[
                dcc.Tabs(
                    id="idx-tabs",
                    value="tab-1",
                    className="custom-tabs",
                    children=[
                        dcc.Tab(label="DEM Comparison", value="tab-1", className="tab", selected_className="tab--selected"),
                        dcc.Tab(label="Track Profile", value="tab-2", className="tab", selected_className="tab--selected"),
                        dcc.Tab(label="ICESat-2 Map", value="tab-3", className="tab", selected_className="tab--selected"),
                        dcc.Tab(label="Best DEM", value="tab-4", className="tab", selected_className="tab--selected"),
                        dcc.Tab(label="CDF Accumulation", value="tab-5", className="tab", selected_className="tab--selected"),
                    ],
                ),
                html.Div(id="idx-tab-content", style={"padding": "10px 12px"}),
            ],
        ),
        html.Div(id="sidebar-backdrop", n_clicks=0),
    ]
)

@callback(
    Output("sidebar-wrap", "className"),
    Input("burger", "n_clicks"),
    Input("collapse", "n_clicks"),
    Input("sidebar-backdrop", "n_clicks"),
    State("sidebar-wrap", "className"),
    prevent_initial_call=True
)
def toggle_sidebar(burger, collapse, backdrop, cls):
    cls = (cls or "").strip()
    changed = dash.ctx.triggered_id
    if changed == "burger":
        return f"{cls} open".strip() if "open" not in cls else cls
    if changed == "sidebar-backdrop":
        return cls.replace("open", "").strip()
    if changed == "collapse":
        return f"{cls} collapsed".strip() if "collapsed" not in cls else cls.replace("collapsed", "").strip()
    return cls
