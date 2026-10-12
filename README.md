
## Adding a raster category to Terracotta (safe procedure)

Never drop files into `data/cogs/` and restart the live container: a restart re-indexes everything inside the
768 MB container, which on 2026-10-12 turned into an OOM restart loop and ~8 minutes of 502. Instead, one category at
a time, indexed in a throwaway container by `services/tc_app/scripts/publish_category.sh`:

```sh
rsync -a <cogs>/ ubuntu@geohydroai.org:gis-sources/tc_pending/<category>/      # flat dir, names [a-z0-9_]+
~/geoai/services/tc_app/scripts/publish_category.sh <category>                  # moves it into data/cogs and indexes it
docker restart geoai-terracotta-1                                               # healthy in ~12 s; unchanged files are skipped
docker exec -i geoai-terracotta-1 python - < ~/geoai/services/tc_app/scripts/raster_versions.py > /tmp/raster-versions.json \
  && sudo install -m 644 /tmp/raster-versions.json /var/www/geoai-front/raster-versions.json
```

Then fetch one tile from inside the container and watch `docker stats geoai-terracotta-1`. The version hash is per
category, so a changed category busts the nginx tile cache for that category only.
