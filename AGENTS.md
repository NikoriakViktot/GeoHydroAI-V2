# AGENTS.md — GeoHydroAI-V2 (Terracotta tiles + FastAPI geoprocessing)

For AI agents and scripts that touch this repo or call its services. This stack sits behind
`https://geohydroai.org`: nginx routes `/tc/` → Terracotta (`services/tc_app`), `/api/` → FastAPI
(`services/api`), `/api/v1/` → the platform (Django, repo `geoai-django`, its own `AGENTS.md`).
Dash (`/dem`) is a legacy explorer.

## Rules

1. **Agents use the platform API, not this service.** Everything an agent needs is exposed through Django
   (`/api/v1/...`), which adds auth, throttling, caching and docs. Call `/api/...` here only when working on
   this repo itself.
2. **Rasters are the source of truth; tiles are pictures.** Terracotta returns colour-mapped 8-bit PNGs. Never
   read a number off a tile pixel — ask `/api/raster/sample` (or, through the platform, `/api/v1/raster/sample`).
3. **Nothing here is per user.** Layers are global reference data (DEMs, HAND, flood masks, LULC …).
   Per-user state (workspaces, scenarios, permissions) lives in the platform.
4. **Do not write under `data/cogs`** or rebuild the Terracotta DB (`services/tc_app/update_terracotta_db.py`)
   without being asked: it is the production tile catalogue.

## Layer keys

Terracotta datasets have two keys, `category/name`, mirroring the file layout
`data/cogs/<category>/<name>_cog.tif` (`clean_name` in `tc_app/update_terracotta_db.py`). Examples:

| Key | What | Units |
|---|---|---|
| `dem/fab_dem`, `dem/copernicus_dem`, … | elevation | m |
| `hand_2000/fab_dem_hand_2000` (also `hand_500/1000/1500`, `hand/`) | Height Above Nearest Drainage, stream threshold in cells | m |
| `flood_scenarios/fab_dem_hand_2000_flood_4m` | depth of water when the river rises 4 m (HAND ≤ 4 m); levels 1–10 m | m |
| `slope_horn/…`, `twi/…`, `lulc/…`, `stream_raster_2000/…` | terrain derivatives, land cover, streams | layer-specific |

- List them: `GET /tc/datasets` (paged; `?category=flood_scenarios`). Metadata, bounds and range:
  `GET /tc/metadata/<category>/<name>`.
- Tile URL (what the SPA uses): `/tc/singleband/<category>/<name>/{z}/{x}/{y}.png?colormap=blues&stretch_range=[0,5]`.
  The stretch is a display choice of the client, not a property of the data.
- Rasters are EPSG:3857, Float32, ~40 m pixels, **nodata = 0** for flood masks (dry land). A nodata pixel is not
  "depth 0".

## FastAPI (`services/api`, served at `/api`)

| Endpoint | Purpose |
|---|---|
| `GET /api/raster/sample?layers=a/b,c/d&lon=&lat=` | **Value under a point.** Up to 4 keys, EPSG:4326. Returns `{lon, lat, crs, values:[{layer, value, nodata, inside}]}`. `nodata: true` = nodata pixel (flood mask: dry) or outside the raster (`inside: false`). Nearest pixel, no interpolation. 404 unknown key, 422 bad key/coords. |
| `POST /api/dem/sample {dem: "dem/fab_dem", points: [[lon,lat],…]}` | Batch version for one layer (needs `ONLINE_SAMPLING=1`). |
| `POST /api/compare/stats`, `/api/compare/profile…` | DEM vs ICESat-2 statistics (online via sampler, offline via Parquet). |
| `GET /api/healthz`, `/api/dbhealth`, `/api/docs` | health, OpenAPI. |

The platform proxies the point query as `GET /api/v1/raster/sample` (anonymous, throttled, cached); the SPA
shows its result as "depth under cursor". When changing the response shape here, update
`geohydro/usecases/raster.py` + `AGENTS.md §5.11` in `geoai-django` and `src/application/api/raster.ts` in
`geoai-frontend-v2`.

Implementation: `infrastructure/raster/cog_sampler.py` (`CogSampler.probe`) opens the COG with rasterio from
`COG_DATA_DIR` (`/app/data/cogs` in compose, mounted read-only into `api` and `terracotta`). Keys are validated
with `^[a-z0-9_]+/[a-z0-9_]+$` and resolved paths must stay inside `COG_DATA_DIR`.

## Running locally

```bash
docker compose up -d terracotta api
curl 'http://localhost:8000/api/raster/sample?layers=flood_scenarios/fab_dem_hand_2000_flood_4m,hand_2000/fab_dem_hand_2000&lon=24.8602&lat=47.8917'
# → depth ≈ 1.31 m, HAND ≈ 2.69 m (4 − 2.69 = 1.31)
```

Without Docker: `cd services/api && COG_DATA_DIR=../../data/cogs PYTHONPATH=. uvicorn main:app --port 8000`
(needs rasterio, and Postgres for the compare/icesat routers).
