# pages/map_flood.py

# pages/map_flood.py
import dash
from dash import html, dcc

dash.register_page(__name__, path="/flood-dem-diif", name="Flood Scenarios", order=98)

def layout():
    return html.Div([
        html.H3("Перенаправляємо на Flood Scenarios…"),
        html.P("Якщо сторінка не відкрилась автоматично, натисніть: "
               + html.A("відкрити вручну", href="/flood_scenarios/").to_plotly_json()["props"]["children"]),
        # Автоперехід (повне перезавантаження сторінки)
        dcc.Location(id="go", href="/flood_scenarios/", refresh=True),
    ], style={"padding":"2rem"})


#
# def _strip_www(u: str) -> str:
#     p = urlparse(u)
#     host = p.netloc.replace("www.", "")
#     return urlunparse((p.scheme or "https", host, p.path, "", "", "")).rstrip("/")
#
#
# TC_BASE = _strip_www(os.getenv("TERRACOTTA_PUBLIC_URL", "https://geohydroai.org/tc"))
# MAPBOX_ACCESS_TOKEN = os.getenv("MAPBOX_ACCESS_TOKEN", "").strip()
# ASSETS_INDEX_PATH = "assets/layers_index.json"
#
# # ---------- Labels & constants ----------
# DEM_LABELS = {
#     "tan_dem": "TanDEM-X",
#     "srtm_dem": "SRTM",
#     "fab_dem": "FABDEM",
#     "copernicus_dem": "Copernicus DEM",
#     "nasa_dem": "NASADEM",
#     "alos_dem": "ALOS",
#     "aster_dem": "ASTER",
# }
# DEM_LIST: List[str] = ["alos_dem", "aster_dem", "copernicus_dem", "fab_dem", "nasa_dem", "srtm_dem", "tan_dem"]
# COLORMAPS = ["viridis", "terrain"]
# MAP_STYLES = {
#     "Outdoors": "mapbox://styles/mapbox/outdoors-v12",
#     "Satellite Streets": "mapbox://styles/mapbox/satellite-streets-v12",
# }
#
# # ---------- Basin (GeoJSON in EPSG:4326) ----------
# try:
#     basin: gpd.GeoDataFrame = get_df("basin")
#     basin = basin.to_crs("EPSG:4326")
#     BASIN_JSON = json.loads(basin.to_json())
#     logger.info("Basin OK: CRS=%s, rows=%d", basin.crs, len(basin))
# except Exception as e:
#     logger.warning("Basin not available: %s", e)
#     BASIN_JSON = None
#
# # ---------- Helpers ----------
# def _fix_path(p: str) -> str:
#     if not p:
#         return p
#     p = p.replace("\\", "/")
#     if p.startswith("/"):
#         return os.path.normpath(p)
#     if p.startswith("data/COG/"):
#         return "/app/data/cogs/" + p.split("data/COG/")[1]
#     if p.startswith("data/cogs/"):
#         return "/app/data/cogs/" + p.split("data/cogs/")[1]
#     if p.startswith("data/"):
#         return "/app/" + p
#     return os.path.normpath(p)
#
# def _parse_level(s: str) -> int:
#     try:
#         return int(str(s).lower().replace("m", "").strip())
#     except Exception:
#         return 0
#
# def build_dem_url(dem_name: str, cmap: str, stretch) -> str:
#     s = f"[{stretch[0]},{stretch[1]}]"
#     return f"{TC_BASE}/singleband/dem/{dem_name}" + "/{z}/{x}/{y}.png" + f"?colormap={cmap}&stretch_range={s}"
#
# def build_flood_url(dem_name: str, hand_name: str, level: str, cmap: str, stretch, pure_blue: bool) -> str:
#     layer = f"{dem_name}_{hand_name}_flood_{level}"
#     s = f"[{stretch[0]},{stretch[1]}]"
#     base = f"{TC_BASE}/singleband/flood_scenarios/{layer}" + "/{z}/{x}/{y}.png"
#     return f"{base}?colormap=custom&colors=0000ff&stretch_range={s}" if pure_blue \
#            else f"{base}?colormap={cmap}&stretch_range={s}"
#
# # ---- deck.gl layer builders (inline renderSubLayers; no custom libraries) ----
# def tile_layer(layer_id: str, url: str, opacity: float = 1.0, visible: bool = True, z: int = 0) -> dict:
#     return {
#         "@@type": "TileLayer",
#         "id": layer_id,
#         "data": url,
#         "visible": visible,
#         "minZoom": 0,
#         "maxZoom": 19,
#         "tileSize": 256,
#         "opacity": opacity,
#         "parameters": {"depthTest": False},
#         "zIndex": z,
#         "renderSubLayers": {
#             "@@function": ["tile", {
#                 "type": "BitmapLayer",
#                 "id": f"{layer_id}-bitmap",
#                 "image": "@@tile.data",
#                 "bounds": "@@tile.bbox",
#                 "opacity": opacity
#             }]
#         },
#     }
#
# def geojson_layer(data: dict, visible: bool = True, z: int = 0) -> dict:
#     return {
#         "@@type": "GeoJsonLayer",
#         "id": "fmap-basin-geojson",
#         "data": data,
#         "visible": visible,
#         "filled": False,
#         "stroked": True,
#         "getFillColor": [30, 144, 255, 0],
#         "getLineColor": [30, 144, 255, 200],
#         "getLineWidth": 2,
#         "lineWidthUnits": "pixels",
#         "zIndex": z,
#     }
#
# import json
#
# def build_spec(map_style, dem_url, flood_url, show_dem, show_flood, show_basin, basin_geojson):
#     layers = []
#     if dem_url:
#         layers.append(tile_layer("fmap-dem-tiles", dem_url, opacity=0.75, visible=show_dem, z=10))
#     if flood_url:
#         layers.append(tile_layer("fmap-flood-tiles", flood_url, opacity=1.0, visible=show_flood, z=20))
#     if basin_geojson:
#         layers.append(geojson_layer(basin_geojson, visible=show_basin, z=30))
#
#     spec = {
#         "mapStyle": map_style if MAPBOX_ACCESS_TOKEN else None,
#         "initialViewState": {"longitude": 25.03, "latitude": 47.8, "zoom": 10, "pitch": 0, "bearing": 0},
#         "layers": layers,
#     }
#     return json.dumps(spec)      # <-- ОБОВ’ЯЗКОВО рядок
# # <-- важливо: рядок, не dict
#
# # ---------- layers_index.json ----------
# layers_index: List[dict] = []
# try:
#     with open(ASSETS_INDEX_PATH, "r") as f:
#         raw_index = json.load(f)
#     items = raw_index if isinstance(raw_index, list) else [raw_index]
#     for rec in items:
#         r = dict(rec)
#         if r.get("path"):
#             r["path"] = _fix_path(r["path"])
#         layers_index.append(r)
#     logger.info("layers_index normalized: %d entries", len(layers_index))
# except Exception as e:
#     logger.warning("Failed to read %s: %s", ASSETS_INDEX_PATH, e)
#     layers_index = []
#
# # Build: DEM_LEVELS[dem] -> ["1m",...]; DEM_LEVEL_TO_HAND[dem][level] -> "hand_2000" or first
# DEM_LEVELS: Dict[str, List[str]] = {}
# DEM_LEVEL_TO_HAND: Dict[str, Dict[str, str]] = {}
# if layers_index:
#     tmp_levels: Dict[str, set] = {}
#     tmp_level2hand: Dict[Tuple[str, str], set] = {}
#     for r in layers_index:
#         if r.get("category") != "flood_scenarios":
#             continue
#         dem, hand, level = r.get("dem"), r.get("hand"), r.get("flood")
#         if not (dem and hand and level):
#             continue
#         tmp_levels.setdefault(dem, set()).add(level)
#         tmp_level2hand.setdefault((dem, level), set()).add(hand)
#     for dem, levels_set in tmp_levels.items():
#         levels_sorted = sorted(levels_set, key=_parse_level)
#         DEM_LEVELS[dem] = levels_sorted
#         DEM_LEVEL_TO_HAND[dem] = {}
#         for lvl in levels_sorted:
#             hands = list(tmp_level2hand.get((dem, lvl), []))
#             DEM_LEVEL_TO_HAND[dem][lvl] = "hand_2000" if "hand_2000" in hands else (hands[0] if hands else "")
#     DEM_LIST = sorted(DEM_LEVELS.keys())
#
# logger.info("DEMs: %s", ", ".join(DEM_LIST))
#
# # ---------- UI ----------
# layout = html.Div([
#     html.H3("Flood Scenarios (deck.gl)"),
#
#     # Map
#     html.Div([
#         dash_deckgl.DashDeckgl(
#             id="fmap-deck",
#             spec=build_spec(
#                 MAP_STYLES["Satellite Streets"],  # map_style
#                 None,  # dem_url
#                 None,  # flood_url
#                 True,  # show_dem
#                 True,  # show_flood
#                 True,  # show_basin
#                 BASIN_JSON
#             ),
#             height=700,
#             mapbox_key=MAPBOX_ACCESS_TOKEN,
#             cursor_position="bottom-right",
#             events=[],
#         )
#     ], style={"position": "relative", "zIndex": 10}),
#
#     # Controls
#     html.Div([
#         html.Div([
#             html.Label("DEM"),
#             dcc.Dropdown(
#                 id="fmap-dem",
#                 options=[{"label": DEM_LABELS.get(d, d), "value": d} for d in DEM_LIST],
#                 value=(DEM_LIST[0] if DEM_LIST else None), style={"width": 220}, clearable=False
#             ),
#         ], style={"display": "inline-block", "marginRight": 12}),
#
#         html.Div([
#             html.Label("Flood level (m)"),
#             dcc.Slider(
#                 id="fmap-flood-level",
#                 min=1, max=10, step=1, value=5,
#                 marks={i: str(i) for i in range(1, 11)},
#                 tooltip={"always_visible": False, "placement": "top"},
#             ),
#         ], style={"display": "inline-block", "width": 260, "marginRight": 12}),
#
#         html.Div([
#             html.Label("DEM Colormap"),
#             dcc.Dropdown(id="fmap-dem-cmap",
#                          options=[{"label": c.capitalize(), "value": c} for c in COLORMAPS],
#                          value="terrain", style={"width": 150}, clearable=False),
#         ], style={"display": "inline-block", "marginRight": 12}),
#
#         html.Div([
#             html.Label("DEM Stretch"),
#             dcc.RangeSlider(id="fmap-dem-stretch", min=0, max=4000, step=50, value=[250, 2200],
#                             marks={0:"0", 1000:"1000", 2000:"2000", 3000:"3000", 4000:"4000"},
#                             tooltip={"always_visible": False, "placement": "top"}),
#         ], style={"display": "inline-block", "width": 360, "marginRight": 12}),
#
#         html.Div([
#             html.Label("Flood Colormap"),
#             dcc.Dropdown(id="fmap-flood-cmap",
#                          options=[{"label": "Blues", "value": "blues"},
#                                   {"label": "Viridis", "value": "viridis"},
#                                   {"label": "Pure Blue", "value": "custom"}],
#                          value="blues", style={"width": 150}, clearable=False),
#         ], style={"display": "inline-block", "marginRight": 12}),
#
#         html.Div([
#             html.Label("Flood Stretch"),
#             dcc.RangeSlider(id="fmap-flood-stretch", min=0, max=10, step=1, value=[0, 5],
#                             marks={i: str(i) for i in range(11)},
#                             tooltip={"always_visible": False, "placement": "top"}),
#         ], style={"display": "inline-block", "width": 260}),
#
#         html.Div([
#             html.Label("Map style"),
#             dcc.Dropdown(id="fmap-map-style",
#                          options=[{"label": k, "value": v} for k, v in MAP_STYLES.items()],
#                          value=MAP_STYLES["Satellite Streets"],
#                          style={"width": 260}, clearable=False),
#         ], style={"display": "inline-block", "marginLeft": 12}),
#
#         html.Div([
#             html.Label("Overlays"),
#             dcc.Checklist(
#                 id="fmap-overlays",
#                 options=[{"label": "Show DEM", "value": "dem"},
#                          {"label": "Show Flood", "value": "flood"},
#                          {"label": "Show Basin", "value": "basin"}],
#                 value=["dem", "flood", "basin"],
#                 inline=True
#             ),
#         ], style={"display": "inline-block", "marginLeft": 18}),
#     ], style={"marginBottom": 12}),
# ])
#
# # ---------- Callbacks ----------
# @callback(
#     Output("fmap-flood-level", "min"),
#     Output("fmap-flood-level", "max"),
#     Output("fmap-flood-level", "marks"),
#     Output("fmap-flood-level", "value"),
#     Input("fmap-dem", "value"),
# )
# def _sync_flood_slider(dem_name: str):
#     levels = DEM_LEVELS.get(dem_name) or []
#     ints = sorted({_parse_level(l) for l in levels})
#     if not ints:
#         # дефолт на всяк випадок
#         return 1, 10, {i: str(i) for i in range(1, 11)}, 5
#     marks = {i: str(i) for i in ints}
#     default = 5 if 5 in ints else ints[0]
#     logger.info("[levels/slider] dem=%s -> range=%s..%s default=%s", dem_name, ints[0], ints[-1], default)
#     return ints[0], ints[-1], marks, default
#
#
# @callback(
#     Output("fmap-deck", "spec"),
#     Input("fmap-dem", "value"),
#     Input("fmap-dem-cmap", "value"),
#     Input("fmap-dem-stretch", "value"),
#     Input("fmap-flood-level", "value"),
#     Input("fmap-flood-cmap", "value"),
#     Input("fmap-flood-stretch", "value"),
#     Input("fmap-map-style", "value"),
#     Input("fmap-overlays", "value"),
# )
# def _update_spec(dem_name, dem_cmap, dem_stretch,
#                  flood_level, flood_cmap, flood_stretch,
#                  map_style, overlays):
#
#     if overlays is None:
#         show_dem = show_flood = show_basin = True
#     else:
#         show_dem = "dem" in overlays
#         show_flood = "flood" in overlays
#         show_basin = "basin" in overlays
#     level_str = f"{int(flood_level)}m" if flood_level is not None else None
#     dem_url = build_dem_url(dem_name, dem_cmap or "terrain", dem_stretch or [250, 2200]) if dem_name else ""
#     flood_url, chosen_hand = "", ""
#     if dem_name and level_str:
#         chosen_hand = (DEM_LEVEL_TO_HAND.get(dem_name, {}) or {}).get(level_str, "")
#         if chosen_hand:
#             flood_url = build_flood_url(
#                 dem_name, chosen_hand, level_str,
#                 flood_cmap or "blues",
#                 flood_stretch or [0, 5],
#                 pure_blue=(flood_cmap == "custom"),
#             )
#
#     logger.info("[spec.in] dem=%s level=%s hand=%s", dem_name, level_str, chosen_hand)
#
#     logger.info("[spec.in] dem=%s cmap=%s stretch=%s level=%s flood_cmap=%s flood_stretch=%s",
#                 dem_name, dem_cmap, dem_stretch, flood_level, flood_cmap, flood_stretch)
#     if dem_name:
#         logger.info("[DEM] visible=%s url=%s", show_dem, dem_url)
#     if flood_level:
#         if flood_url:
#             layer_name = f"{dem_name}_{chosen_hand}_flood_{flood_level}"
#             logger.info("[FLOOD] layer=%s visible=%s url=%s", layer_name, show_flood, flood_url)
#         else:
#             reason = ("no matching hand" if not chosen_hand else "url not built")
#             logger.info("[FLOOD] not shown (%s) dem=%s level=%s", reason, dem_name, flood_level)
#     logger.info("[MAP] style=%s basin=%s", map_style, show_basin)
#
#
#     return build_spec(map_style, dem_url or None, flood_url or None,
#                       show_dem, show_flood, show_basin, BASIN_JSON)# JSON string
