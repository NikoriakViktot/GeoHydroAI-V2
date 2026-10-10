#!/usr/bin/env bash
# Smoke test of every public route of geohydroai.org behind geoai-nginx: expected status per path.
# 401 = the route exists but needs a key / login; 403/404 where that is the intended answer.
#   services/nginx/tests/smoke.sh [base-url]
set -uo pipefail
BASE=${1:-https://geohydroai.org}
fail=0
check() {  # check <expected codes, |-separated> <path> [what]
  local want=$1 path=$2 what=${3:-}
  local got; got=$(curl -s -o /dev/null -w "%{http_code}" --max-time 30 "$BASE$path")
  if [[ "|$want|" == *"|$got|"* ]]; then printf "ok    %s  %-50s %s\n" "$got" "$path" "$what"
  else printf "FAIL  %s  %-50s %s (want %s)\n" "$got" "$path" "$what" "$want"; fail=1; fi
}
check_header() {  # check_header <path> <header regex>
  if curl -sI --max-time 30 "$BASE$1" | grep -qiE "$2"; then printf "ok    hdr  %-50s %s\n" "$1" "$2"
  else printf "FAIL  hdr  %-50s %s\n" "$1" "$2"; fail=1; fi
}

# redirects / TLS
got=$(curl -s -o /dev/null -w "%{http_code}" "http://${BASE#https://}/")
[[ $got == 301 ]] && echo "ok    301  http -> https" || { echo "FAIL  $got  http -> https"; fail=1; }
check 200 "/"                        "SPA v1"
check 200 "/app/"                    "SPA v2"
check 301 "/app"                     "-> /app/"
check 200 "/flood-map"                "SPA deep link (try_files fallback)"
check 404 "/.env"                    "hidden files blocked"
check 404 "/.git/config"             "hidden files blocked"
check 200 "/__whoami"
# geoai FastAPI (api:8000) and Dash (dash:8050) — catches stale upstream IPs (502)
check 200 "/api/docs"                "geoai-api"
check "200|404" "/api/health"        "geoai-api answers (not 502)"
check 200 "/dem/"                    "dash"
check 302 "/dem"                     "-> /dem/"
check 200 "/dem/dashboard"            "dash dashboard page"
# platform-api (Django)
check 200 "/api/v1/health"           "platform-api"
check 403 "/api/v1/internal/callbacks/hydro/snapshot" "internal blocked"
check 200 "/api/schema"
check 200 "/api/v1/flood-scenarios?format=geojson" "public flood scenarios"
check 200 "/api/v1/flood-scenarios/at?lat=48.0464&lon=24.8943"
check 401 "/api/v1/hydro/stations"   "hydro proxy needs login"
check 401 "/api/v1/icesat2/products" "icesat2 proxy exists, needs login (not 404/502)"
# hydro-api
check 401 "/v1/hydro"                "hydro-api needs key"
check 403 "/v1/hydro/admin/x"        "admin blocked"
check 200 "/llms.txt"
check 200 "/llms-full.txt"
# vector tiles (Martin)
check 200 "/tiles/catalog"
for s in rivers river_system lakes basins soils posts; do check 200 "/tiles/$s/6/37/21" "MVT $s"; done
check 200 "/tiles/sea_waters/6/37/22"  "MVT sea_waters (Black Sea tile)"
check 204 "/tiles/sea_waters/6/37/21"  "empty tile over land = 204, not an error"
check 200 "/tiles/swb_boundaries/9/298/172" "MVT swb_boundaries"
check 200 "/tiles/apsfr_reaches/5/18/10"  "official APSFR river reaches (DSNS)"
check 200 "/tiles/apsfr_points/7/73/43"   "official APSFR territories (DSNS)"
check 200 "/tiles/dsns_flood_hazard/11/1157/703" "DSNS flood hazard map (Opir)"
check 200 "/tiles/catchments/10/582/355" "post catchments (42194 Yablunytsia)"
check_header "/tiles/rivers/6/37/21" "^cache-control: public, max-age=3600"
# terracotta, catalog, reports
check 200 "/tc/keys"                 "terracotta"
check 200 "/tc/metadata/curve_number/cn2_grid" "CN II raster registered"
check 200 "/tc/singleband/curve_number/cn2_grid/6/37/21.png?colormap=rdylgn_r&stretch_range=%5B40,95%5D" "CN II raster tile"
check 301 "/catalog"                 "-> /catalog/"
check 200 "/catalog/health"          "catalog-api"
check 403 "/reports/"                "no directory listing"
# python-course vhost
got=$(curl -s -o /dev/null -w "%{http_code}" --max-time 30 https://python-course-viktor-nikoriak.org/)
[[ $got =~ ^(200|301|302)$ ]] && echo "ok    $got  python-course" || { echo "FAIL  $got  python-course"; fail=1; }

[[ $fail == 0 ]] && echo "ALL OK" || { echo "SOME CHECKS FAILED"; exit 1; }
