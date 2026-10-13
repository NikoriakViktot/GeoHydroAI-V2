# geoai-api v1 (`services/api`) — Specification (v1, legacy)

## 1. Purpose and status

FastAPI service of the v1 platform (`root_path=/api`, container `geoai-api-1`, 1 GB,
`127.0.0.1:8000`). In v2 **one endpoint still matters**: `GET /api/raster/sample`, the value
under the cursor, which the platform proxies as `GET /api/v1/raster/sample` (anonymous,
throttled 600/min, cached 300 s). Everything else (DEM-vs-ICESat-2 compare, ICESat queries,
`/dem/sample`) is v1 tooling with no v2 consumer; `dash` v1 (`/dem/`) had 0 requests in the 72 h
before 2026-10-13.

Plan: move `raster/sample` into the platform (Django `geohydro/usecases/raster.py` already owns
the contract) or into hydro-api, then retire this container and dash (−2.5 GB of memory limits).

## 2. Contract: `GET /api/raster/sample`

- Query: `layers=<category>/<name>[,…]` (1–4 keys, `^[a-z0-9_]+/[a-z0-9_]+$`), `lon`, `lat`
  (EPSG:4326).
- Response: `{lon, lat, crs: "EPSG:4326", values: [{layer, value, nodata, inside}]}`; nearest
  pixel; `nodata: true` on a flood mask means dry land, not depth 0.
- Errors: 422 bad key or count; 404 unknown layer.
- Implementation `infrastructure/raster/cog_sampler.py` (`CogSampler.probe`): opens
  `<COG_DATA_DIR>/<category>/<name>_cog.tif` (fallback `<name>.tif`), resolved path must stay
  inside `COG_DATA_DIR` (traversal guard), reprojects the point, reads a 1×1 window.
- Change the shape only together with geoai-django `geohydro/usecases/raster.py` + AGENTS.md
  §5.11 and geoai-frontend-v2 `src/application/api/raster.ts`.

Other routes: `/api/healthz` (compose healthcheck), `/api/dbhealth`, `POST /api/dem/sample`
(needs `ONLINE_SAMPLING=1`), `POST /api/compare/{stats,profile}`, `POST /api/icesat/query`,
`GET /api/icesat/{id}` (stub). `routers/filters.py` is not mounted.

## 3. Configuration

`POSTGRES_URL`, `ALLOWED_ORIGINS` (`*`), `LOG_LEVEL`, `ONLINE_SAMPLING`, `SLOPE_LAYER`,
`HAND_LAYER`, `PARQUET_FILE`, `COG_DATA_DIR`; `TERRACOTTA_URL` and `DEFAULT_DEMS` are read by
nothing live. Depends on `postgis` (healthy) and `terracotta` (healthy); the postgis image no
longer builds (apt.postgresql.org dropped bullseye), so the deploy rebuilds only terracotta.

## 4. Invariants and gaps

MUST: keep `raster/sample` read-only, path-guarded and fast (one window read); keep
`/healthz` for the healthcheck.
Gaps: no auth and CORS `*` (acceptable only because nginx and the platform throttle it); runs as
root; the dependency on `postgis` ties a 1.5 GB database container to a 1×1 raster read — the
sampler needs no database.

## 5. Verification

```bash
curl -s 'https://geohydroai.org/api/v1/raster/sample?layers=dem_ua/fab_dem&lon=25.94&lat=48.29'
docker logs --since 72h geoai-api-1 2>&1 | grep -c raster/sample
```
