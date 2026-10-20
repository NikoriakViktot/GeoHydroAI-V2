# nginx edge (`services/nginx/nginx.conf`) — Specification (v1)

## 1. Purpose

The single public entry point of geohydroai.org (80/443, container `geoai-nginx`, 256 MB):
TLS termination (Let's Encrypt, renewed by `certbot-renewer` every 12 h, config reloaded every
6 h), static frontend, and the **routing + caching + rate-limiting layer** in front of every
service. It also fronts a second domain (python-course) for an unrelated project.

> This file is the source of truth for "which path goes where". The repos of the services
> (geoai-django AGENTS.md, data_dnipro_h_q CLAUDE.md) point here.
> The config is bind-mounted as a directory (`./services/nginx:/etc/nginx/geoai:ro`) so that
> `git reset --hard` on deploy does not pin a stale inode.

## 2. Location map (443)

| Path | Upstream | Limits / cache |
|---|---|---|
| `/` (+ `/index.html`, `/assets/`) | static `/var/www/geoai-front/current` (SPA v2; `/app/` = legacy v2 path) | assets immutable 7 d / 1 y |
| `= /raster-versions.json` | static | no-cache |
| `^~ /api/v1/internal/` | **403** | defence in depth for Django callbacks |
| `^~ /api/v1/`, `= /api/schema`, `= /api/docs` | `platform-api:8000` | body 50 MB (yearbook uploads), read 120 s; nginx's own 502/504 → `@api_busy` (503 JSON `{detail, code: "server_busy"}`, `Retry-After: 30`) |
| `^~ /api/v1/labs/` | `platform-api:8000` | body 50 MB, **buffered** by nginx (book uploads from slow uplinks must not hold a gunicorn worker past its 60 s timeout) |
| `^~ /api/` | `api:8000` (geoai-api v1: `/raster/sample`, `/healthz`, legacy compare/icesat) | server defaults (2 GB body, 3600 s) |
| `^~ /v1/hydro/admin/` | **403** | |
| `^~ /v1/hydro`, `~ ^/llms(-full)?\.txt$` | `hydro-api:8000` | body 2 MB, read 120 s |
| `^~ /tiles/` | `hydro-tiles:3000` (Martin), `/tiles` stripped | `tiles_cache` 10 min (200/204), bypass on `Authorization`/`ghai_refresh`, stale on error |
| `^~ /tc/singleband/`, `/tc/rgb/` | `terracotta:5000` | `tc_cache` 30 d, 404 1 min, lock, 40 r/s burst 80 |
| `^~ /tc/` | terracotta metadata | no cache |
| `^~ /catalog/` | `catalog-api:8095` (only `v1/(cards\|facets\|release)`, `health`; GET/HEAD) | 10 r/s burst 30, body 16 k, read 30 s, `X-API-Key` injected from `/etc/nginx/secrets/catalog/*.conf` |
| `^~ /dem/`, `/_dash-*` | `dash:8050` (legacy v1 explorer) | |
| `/reports/` | static `/var/www/geoai-reports` | |
| `= /__whoami` | debug 200 | remove in production |

Hidden files (`/\.` except `.well-known`) → 404. Port 80: ACME challenge, else 301.

```mermaid
flowchart LR
  B[Browser] -->|443| N[geoai-nginx]
  N -->|/ static| F[/var/www/geoai-front]
  N -->|/api/v1| P[platform-api]
  N -->|/api| A1[geoai-api v1]
  N -->|/v1/hydro, /llms.txt| H[hydro-api]
  N -->|/tiles| M[hydro-tiles Martin]
  N -->|/tc| T[terracotta]
  N -->|/catalog + X-API-Key| K[catalog-api]
  N -->|/dem| D[dash v1]
  N -->|python-course domain| PC[web:9000]
```

## 3. Invariants

- Every service port is `expose`d or bound to `127.0.0.1`; only nginx publishes 80/443.
- `/api/v1/internal/` and `/v1/hydro/admin/` are never reachable from the Internet.
- Tile caches key on the full URI incl. `?v=`; a new `raster-versions.json` is the cache-bust.
- `/api/v1/` never shows an nginx HTML error page: when platform-api is down or too slow (a model run, a data load
  on the 2-vCPU host) the SPA gets `503 {code: "server_busy"}` and shows a «server busy» banner. Django's own
  responses are never rewritten (no `proxy_intercept_errors`).
- `nginx -t` runs before every reload in `deploy_aws.yml`; a bad config keeps the old one live.
- Secrets (catalog key) enter only through the mounted `secrets/catalog` directory, never the
  config in git.

## 4. Known gaps (2026-10-13)

1. Server-level `client_max_body_size 2048M` and `proxy_read_timeout 3600s` apply to every
   location without its own limit (`/api/`, `/dem/`, `/tc/` metadata, `/reports/`): set 2 MB and
   60 s at server level and raise them only where uploads happen.
2. No security headers: add `Strict-Transport-Security` (after confirming all subdomains are
   HTTPS), `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY` (or CSP
   `frame-ancestors`), `Referrer-Policy: strict-origin-when-cross-origin`, `server_tokens off`.
3. `/api/docs` goes to Django, so FastAPI v1 docs are unreachable (fine) but confusing — drop the
   v1 `docs_url` or document it.
4. `= /__whoami` debug location is public.
5. The python-course domain shares the container and the `python_course` network with the
   platform; moving that project off the host removes the coupling.

## 5. Verification

```bash
docker exec geoai-nginx nginx -c /etc/nginx/geoai/nginx.conf -t
curl -sI https://geohydroai.org/api/v1/internal/x | head -1          # 403
curl -sI https://geohydroai.org/v1/hydro/admin/jobs | head -1        # 403
curl -sI 'https://geohydroai.org/tc/singleband/dem_ua/fab_dem/6/37/22.png?colormap=gray&stretch_range=[0,2000]&v=x' | grep -i -E 'cache-control|x-cache'
```
