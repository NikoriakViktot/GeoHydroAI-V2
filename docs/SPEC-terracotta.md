# Terracotta raster tiles (`services/tc_app`) — Specification (v1)

## 1. Purpose

Serves the **public raster overlays** of GeoHydroAI (DEMs, HAND, flood masks, slope, streams,
FFPI, national `*_ua` layers) as colour-mapped PNG tiles from Cloud-Optimised GeoTIFFs in
`data/cogs/<category>/<name>_cog.tif`. Terracotta 0.8.2 under gunicorn (2 gevent workers),
container `geoai-terracotta-1`, 768 MB, `expose 5000`, reachable only through nginx at `/tc/`.

> Tiles are pictures: no client reads a value off a pixel. The value under the cursor comes
> from `GET /api/v1/raster/sample` (platform → geoai-api v1 `CogSampler`), which reads the same
> COG.
> The live container never indexes new categories (OOM loop of 2026-10-12); indexing runs in a
> throwaway container (`scripts/publish_category.sh`).

## 2. URL contract (behind nginx `/tc/`)

| Path | Cache | Notes |
|---|---|---|
| `/tc/singleband/<category>/<name>/{z}/{x}/{y}.png?colormap=&stretch_range=[a,b]` | nginx `tc_cache` 30 d (200), 1 min (404); `?v=<hash>` → `immutable` 1 y, else 1 d | the SPA's only tile URL; `rasterTileLayer` appends `v` from `/raster-versions.json` |
| `/tc/rgb/…` | same | unused by the SPA |
| `/tc/datasets?category=`, `/tc/metadata/<category>/<name>`, `/tc/keys` | none | live metadata |
| `/tc/health`, `/tc/apidoc`, `/tc/swagger.json` | none | |

Rate limit 40 r/s per IP, burst 80 → 429. Keys: `category`, `name` (`clean_name` strips
`_cog.tif`, `.tif`, `_utm32635`). Rasters EPSG:3857, Float32, ~40 m; flood masks nodata 0 =
dry land. Reprojection nearest, resampling linear (hard-coded in `init_tc_server.py`).

## 3. Indexer (`update_terracotta_db.py`)

- Walks `COG_DATA_DIR`; category = relative directory, `uncategorized` at the root.
- Creates the SQLite driver `TC_DB` with keys `[category, name]` if missing.
- Rasters above `TC_LARGE_PIXELS` (50 M px) get metadata from a ≤ 4096² overview (the fix for
  the 2026-10-12 OOM).
- Skips unchanged files (`size:mtime_ns` in `indexed.json` next to the DB); `TC_REINDEX=1`
  forces a full pass. Writes `metadata.json` for Dash.
- Runs on every container start (`entrypoint.sh`) — fast when nothing changed.

## 4. Publishing a category (the only supported write path)

1. rsync COGs to `~/gis-sources/tc_pending/<category>/` (flat, `[a-z0-9_]+`, `<name>_cog.tif`).
2. `scripts/publish_category.sh <category>`: moves the dir into `~/geoai/data/cogs/<category>`
   (refuses an existing one) and indexes in `docker run --rm --memory 1500m` with the
   `geoai_tcdb` volume.
3. `docker restart geoai-terracotta-1` (healthy in ~12 s).
4. `docker exec -i geoai-terracotta-1 python - < scripts/raster_versions.py > /tmp/raster-versions.json && sudo install -m 644 … /var/www/geoai-front/raster-versions.json`
   — per-category sha1 of `name:size:mtime_ns`, served at `/raster-versions.json` (no-cache).
5. Smoke-test one tile from inside the container; watch `docker stats`.

Replacing an existing category (e.g. the border-clipped `*_ua` layers): move the old dir aside
first, publish, then remove the old dir; the version hash changes and clients refetch.

## 5. Configuration

`COG_DATA_DIR` (`/app/data/cogs`), `TC_DB` (`/app/tc_db/terracotta.sqlite`), `TC_PORT` 5000,
`TC_HOST`, `TC_LARGE_PIXELS`, `TC_REINDEX`; `RESAMPLING_METHOD` is set in compose but ignored.
Script-only: `GEOAI_ROOT`, `TC_PENDING`. Volumes: `./data/cogs` ro, `tcdb`, `tc_tmp:/tmp`.

## 6. Invariants and gaps

MUST: keep the live container's memory flat (metadata from overviews, index elsewhere); keep
`raster-versions.json` in step with the DB after every publish; serve only `[a-z0-9_]+` keys.
MUST NOT: write into `data/cogs` from any service; restart the live container with unindexed
large rasters present.
Gaps: no healthcheck on tile rendering (only `/health`); `TILE_PATH_TEMPLATE` in `config.py` is
a leftover; the container runs as root with the COG dir mounted ro (acceptable, but `USER` is
cheap); the deploy rebuilds the image on the production host.

## 7. Verification

```bash
docker exec -i geoai-terracotta-1 python - <<'PY'
import urllib.request; print(urllib.request.urlopen("http://127.0.0.1:5000/singleband/dem_ua/fab_dem/6/37/22.png?colormap=gray&stretch_range=[0,2000]").status)
PY
curl -s https://geohydroai.org/raster-versions.json | python3 -m json.tool | head
```
