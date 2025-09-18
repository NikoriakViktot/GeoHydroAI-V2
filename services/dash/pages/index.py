# pages/index.py
import dash
from dash import html, dcc, callback, Output, Input, State
from layout.sidebar import sidebar

dash.register_page(__name__, path="/", name="Dashboard", title="GeoHydroAI | Dashboard", order=0)

layout = html.Div(
    id="layout",
    children=[
        html.Button("☰", id="burger", n_clicks=0, className="hamburger"),
        html.Div(id="sidebar-wrap", children=[
            html.Button("⮜", id="collapse", n_clicks=0, className="collapse-btn"),
            sidebar,  # без position:fixed всередині (див. utils/style.py)
        ]),
        html.Div(
            id="content",
            children=[
                # Stores, які можуть знадобитися колбекам
                dcc.Store(id="cdf-store", storage_type="session"),
                dcc.Tabs(
                    id="idx-tabs",
                    value="tab-1",
                    className="custom-tabs",
                    children=[
                        dcc.Tab(label="Comparison", value="tab-1", className="tab", selected_className="tab--selected"),
                        dcc.Tab(label="Profile",    value="tab-2", className="tab", selected_className="tab--selected"),
                        dcc.Tab(label="Map",        value="tab-3", className="tab", selected_className="tab--selected"),
                        dcc.Tab(label="Best DEM",   value="tab-4", className="tab", selected_className="tab--selected"),
                        dcc.Tab(label="CDF Accumulation", value="tab-5", className="tab", selected_className="tab--selected"),
                    ],
                ),
                html.Div(id="idx-tab-content", style={"padding": "10px 12px"}),
            ],
        ),
        html.Div(id="sidebar-backdrop", n_clicks=0),
    ],
)

# ЄДИНИЙ локальний колбек – керування класами сайдбару (open/collapsed)
@callback(
    Output("sidebar-wrap", "className"),
    Input("burger", "n_clicks"),
    Input("collapse", "n_clicks"),
    Input("sidebar-backdrop", "n_clicks"),
    State("sidebar-wrap", "className"),
    prevent_initial_call=True,
)
def toggle_sidebar(burger, collapse, backdrop, cls):
    cls = cls or ""
    changed = dash.ctx.triggered_id
    if changed == "burger":
        return (cls + " open").strip() if "open" not in cls else cls
    if changed == "sidebar-backdrop":
        return cls.replace("open", "").strip()
    if changed == "collapse":
        return (cls + " collapsed").strip() if "collapsed" not in cls else cls.replace("collapsed", "").strip()
    return cls
