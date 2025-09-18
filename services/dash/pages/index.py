# pages/index.py
import dash
from dash import html, dcc, callback, Output, Input, State
import plotly.graph_objects as go
from layout.sidebar import sidebar
# якщо маєш PageShell — можна використати, але тут зробимо без нього
from main_tab import render_main_tab  # функція очікує кілька фігур/даних :contentReference[oaicite:1]{index=1}

dash.register_page(__name__, path="/", name="Dashboard", title="GeoHydroAI | Dashboard", order=0)

layout = html.Div(
    id="layout",
    children=[
        # Кнопки керування сайдбаром (CSS у тебе вже готове)
        html.Button("☰", id="burger", n_clicks=0, className="hamburger"),
        html.Div(id="sidebar-wrap", children=[
            html.Button("⮜", id="collapse", n_clicks=0, className="collapse-btn"),
            sidebar
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
                html.Div(id="idx-tab-content", style={"padding": "10px 12px"})
            ],
        ),
        html.Div(id="sidebar-backdrop", n_clicks=0)  # затемнення на мобайлі
    ]
)

# 1) Вставляємо контейнер під активну вкладку (для табу-1 це `dashboard-main`)
@callback(Output("idx-tab-content", "children"), Input("idx-tabs", "value"))
def render_tab_body(tab):
    if tab == "tab-1":
        # Контейнер, у який малюватиме наступний колбек
        return html.Div([html.H1("DEM Comparison"), html.Div(id="dashboard-main")])
    if tab == "tab-2":
        return html.Div([html.H3("Track Profile (WIP)")])
    if tab == "tab-3":
        return html.Div([html.H3("ICESat-2 Tracks Map (WIP)")])
    if tab == "tab-4":
        return html.Div([html.H3("Best DEM (WIP)")])
    if tab == "tab-5":
        return html.Div([html.H3("CDF Accumulation (WIP)"), html.Div(id="cdf-content")])
    return html.Div()

# 2) Мінімальний контент для DEM Comparison, щоб щось було видно одразу
@callback(
    Output("dashboard-main", "children"),
    Input("idx-tabs", "value"),          # перемкнувся на таб → покажи щось
    Input("apply_filters_btn", "n_clicks")  # натиснув фільтри → онови
)
def fill_dem_comparison(tab, n_clicks):
    if tab != "tab-1":
        return dash.no_update

    # Плейсхолдери — просто щоб перевірити рендер
    hist_fig = go.Figure(go.Histogram(x=[1,2,2,3,3,3,4,4,5]))
    box_fig  = go.Figure(go.Box(y=[1,2,3,2,5,3,2,4,3]))
    bar_fig  = go.Figure(go.Bar(x=["ALOS","FAB","SRTM"], y=[0.9, 0.7, 0.5]))
    table_rows = [
        {"DEM":"ALOS", "NMAD":0.85, "RMSE":1.20},
        {"DEM":"FAB",  "NMAD":0.65, "RMSE":0.95},
        {"DEM":"SRTM", "NMAD":1.10, "RMSE":1.60},
    ]
    columns = [{"name":c, "id":c} for c in ["DEM","NMAD","RMSE"]]
    filters_summary = "Demo-state: slope 0–60°, HAND off"
    dem = "fab_dem"
    title = "Metrics by DEM (sample)"

    return render_main_tab(
        hist_fig, box_fig, bar_fig,
        table_rows, columns, title,
        dem, filters_summary
    )
