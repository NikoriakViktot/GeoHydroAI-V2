#!/bin/sh
# Publish ONE raster category to the production Terracotta without touching the live container.
#
#   usage (on the server):  ~/geoai/services/tc_app/scripts/publish_category.sh <category>
#   before:  rsync the COGs to ~/gis-sources/tc_pending/<category>/  (flat dir, <name>_cog.tif or <name>.tif,
#            names [a-z0-9_]+ — the frontend's tcLayerKey and /raster/sample accept nothing else)
#   after:   docker restart geoai-terracotta-1 ; regenerate raster-versions.json (README "Adding a raster category")
#
# Why a throwaway container: on 2026-10-12 dropping national COGs into data/cogs and restarting put the live
# Terracotta (768 MB) into an OOM restart loop and the site returned 502 for ~8 minutes. Indexing here runs
# update_terracotta_db.py in its own container with its own memory limit; the live server keeps serving.
set -eu
c=${1:?category}
root=${GEOAI_ROOT:-$HOME/geoai}
pending=${TC_PENDING:-$HOME/gis-sources/tc_pending}
if [ ! -d "$pending/$c" ]; then echo "no $pending/$c" >&2; exit 1; fi
if [ -e "$root/data/cogs/$c" ]; then echo "$root/data/cogs/$c already exists: remove or merge it first" >&2; exit 1; fi
mv "$pending/$c" "$root/data/cogs/$c"
docker run --rm --memory 1500m --memory-swap 1500m --name "tc-index-$c" \
  -e COG_DATA_DIR=/app/data/cogs -e TC_DB=/app/tc_db/terracotta.sqlite -e PYTHONPATH=/app/tc_app -w /tmp \
  -v "$root/data/cogs:/app/data/cogs" -v geoai_tcdb:/app/tc_db \
  --entrypoint python geoai-terracotta /app/tc_app/update_terracotta_db.py 2>&1 | grep -v '^$' | tail -20
echo "indexed $c; now: docker restart geoai-terracotta-1 && regenerate raster-versions.json"
