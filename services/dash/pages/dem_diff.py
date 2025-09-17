# pages/dem_diff.py

import dash
from dash import html

dash.register_page(__name__, path="/dem-diff", name="Dashboard", order=1)

layout = html.Div([
    html.H1("Це твій Dashboard! 🎉"),
    html.P("Все працює!"),
])


# import dash
# import dash_leaflet as dl
# from dash import html, dcc, callback, Output, Input, State
# import json
# import os
# import numpy as np
# import geopandas as gpd
# from collections import defaultdict
#
# from utils.dem_tools import (
#     compute_dem_difference,
#     save_temp_diff_as_cog,
#     save_colorbar_png,
#     plot_histogram,
#     calculate_error_statistics,
# )
#
# dash.register_page(__name__, path="/dem-diff", name="DEM Diff (Tiles)", order=1)
#
# # --- Завантаження меж басейну (geojson) ---
# basin_gdf = gpd.read_file("data/basin_bil_cher_4326.gpkg").to_crs("EPSG:4326")
# basin = json.loads(basin_gdf.to_json())
#
# # --- Завантаження індексу всіх шарів ---
# LAYERS_INDEX_PATH = "data/layers_index.json"
# COLORBAR_PATH = "assets/diff_colorbar.png"
# TEMP_DIFF_DIR = "/tmp"
#
# with open(LAYERS_INDEX_PATH, "r") as f:
#     layers_index = json.load(f)
#
# # --- Групування за DEM та категорією для фільтрів ---
# by_dem = defaultdict(list)
# categories = set()
# for l in layers_index:
#     by_dem[l["dem"]].append(l)
#     categories.add(l["category"])
#
# dem_list = sorted(by_dem.keys())
# category_list = sorted(categories)
#
# # --- Layout ---
# sidebar = html.Div([
#     html.H4("Фільтри", style={"color": "#EEEEEE"}),
#     dcc.Dropdown(id="dem1_select", options=[{"label": d, "value": d} for d in dem_list], placeholder="DEM 1"),
#     dcc.Dropdown(id="dem2_select", options=[{"label": d, "value": d} for d in dem_list], placeholder="DEM 2"),
#     dcc.Dropdown(id="category_select", options=[{"label": c, "value": c} for c in category_list], placeholder="Категорія"),
#     html.Button("Порахувати різницю", id="run_analysis_btn"),
# ], style={"width": "350px", "padding": "12px"})
#
# layout = html.Div([
#     html.H3("DEM Difference Analysis (універсальний TileServer OLAP)"),
#     html.Div([sidebar]),
#     html.Div([
#         dl.Map([
#             dl.TileLayer(id="dem-tile"),
#             dl.GeoJSON(data=basin, id="basin", options={"style": {"color": "#006aff", "weight": 2, "fill": False}})
#         ], id="leaflet-map", center=[48.5, 32.5], zoom=7, style={'height': '60vh', 'width': '100%'})
#     ]),
#     html.Div([
#         html.Img(src="/assets/diff_colorbar.png", style={"height": "200px"}),
#         html.Img(id="diff-hist", style={"height": "180px", "marginLeft": "40px"})
#     ], style={"display": "flex", "alignItems": "center", "marginTop": "30px"}),
#     html.Div(id="diff-stats", style={"marginTop": 20, "fontFamily": "monospace"})
# ])
#
# # --- CALLBACK: аналіз різниці DEM через TileServer ---
# @callback(
#     Output("dem-tile", "url"),
#     Output("diff-hist", "src"),
#     Output("diff-stats", "children"),
#     Input("run_analysis_btn", "n_clicks"),
#     State("dem1_select", "value"),
#     State("dem2_select", "value"),
#     State("category_select", "value"),
#     prevent_initial_call=True
# )
# def run_dem_diff(n_clicks, dem1, dem2, category):
#     if not dem1 or not dem2 or dem1 == dem2:
#         return dash.no_update, dash.no_update, "Оберіть різні DEM!"
#
#     # --- Знаходимо шляхи для DEM (можеш врахувати category, наприклад 'original') ---
#     path1 = next((l["path"] for l in by_dem[dem1] if l["category"] == category), None)
#     path2 = next((l["path"] for l in by_dem[dem2] if l["category"] == category), None)
#
#     if not path1 or not path2:
#         return dash.no_update, dash.no_update, "DEM не знайдено у layers_index!"
#
#     # --- Обчислюємо різницю ---
#     try:
#         diff, ref_dem = compute_dem_difference(path1, path2)
#     except Exception as e:
#         return dash.no_update, dash.no_update, f"Помилка при обчисленні різниці: {e}"
#
#     try:
#         diff_cog_path = save_temp_diff_as_cog(diff, ref_dem, prefix="demdiff_")
#     except Exception as e:
#         return dash.no_update, dash.no_update, f"Помилка при збереженні COG: {e}"
#
#     tile_url = f"/tc/tiles/{os.path.basename(diff_cog_path)}/{{z}}/{{x}}/{{y}}.png"
#
#     # --- Гістограма і статистика ---
#     try:
#         q1, q99 = np.nanpercentile(diff, [1, 99])
#         hist_img = plot_histogram(diff, clip_range=(q1, q99))
#     except Exception as e:
#         hist_img = None
#
#     try:
#         stats = calculate_error_statistics(diff)
#         stats_table = html.Table([
#             html.Tr([html.Th(k), html.Td(f"{v:.3f}" if isinstance(v, float) and not np.isnan(v) else v)])
#             for k, v in stats.items()
#         ], style={"color": "#EEE", "backgroundColor": "#181818", "fontFamily": "monospace", "marginTop": "15px"})
#     except Exception as e:
#         stats_table = f"Помилка при підрахунку статистики: {e}"
#
#     return tile_url, hist_img, stats_table
