#!/usr/bin/env bash
# Restore test of the newest geohydro dump FROM S3 into an isolated PostGIS container; writes /data/backups/pg/.restore_verified on success.
set -uo pipefail
B=geo-hydro-ai-data-116344514823-us-east-1-an; R=us-east-1; DEST=/data/backups/pg; W=/data/backups/restore_test; mkdir -p "$W"
key=$(aws s3 ls "s3://$B/backups/pg/" --region $R | awk '{print $4}' | grep '^geohydro_' | sort | tail -1)
echo "[$(date -u +%T)] dump $key"
aws s3 cp "s3://$B/backups/pg/$key" "$W/$key" --region $R --only-show-errors || { echo "download failed"; exit 1; }
echo "[$(date -u +%T)] downloaded $(du -h "$W/$key" | cut -f1); listing"
docker run --rm -v "$W":/w postgis/postgis:16-3.5 pg_restore -l "/w/$key" | wc -l | sed 's/^/toc entries: /'
docker rm -f pg_restore_test >/dev/null 2>&1
docker run -d --name pg_restore_test -e POSTGRES_PASSWORD=restore -e POSTGRES_DB=geohydro --memory 1500m --cpus 1 -v "$W":/w postgis/postgis:16-3.5 >/dev/null
# the postgis entrypoint starts a temporary server for init scripts and restarts: wait for the SECOND "ready to accept"
for i in $(seq 1 90); do [ "$(docker logs pg_restore_test 2>&1 | grep -c "ready to accept connections")" -ge 2 ] && break; sleep 2; done
sleep 3; docker exec pg_restore_test pg_isready -U postgres || { echo "postgres not ready"; docker logs --tail 20 pg_restore_test; exit 1; }
echo "[$(date -u +%T)] restoring"
docker exec pg_restore_test pg_restore -U postgres -d geohydro -j 1 --no-owner --no-privileges "/w/$key" 2> "$W/restore.err"; rc=$?
echo "[$(date -u +%T)] pg_restore exit $rc, stderr lines $(wc -l < "$W/restore.err")"
q='SELECT schemaname||'"'"'.'"'"'||relname, n_live_tup FROM pg_stat_user_tables ORDER BY n_live_tup DESC LIMIT 6'
echo "-- restored"; docker exec pg_restore_test psql -U postgres -d geohydro -Atc "$q"
echo "-- live";     docker exec geoai-postgis-1  psql -U postgres -d geohydro -Atc "$q"
cnt() { docker exec "$1" psql -U postgres -d geohydro -Atc "SELECT count(*) FROM $2" 2>/dev/null; }
ok=1; checked=0
for t in $(docker exec geoai-postgis-1 psql -U postgres -d geohydro -Atc "SELECT schemaname||'.'||relname FROM pg_stat_user_tables ORDER BY n_live_tup DESC LIMIT 6"); do
  a=$(cnt pg_restore_test "$t"); b=$(cnt geoai-postgis-1 "$t"); echo "$t restored=${a:-ERR} live=${b:-ERR}"
  if [ -n "$a" ] && [ -n "$b" ] && [ "$a" = "$b" ]; then checked=$((checked + 1)); else ok=0; fi
done
[ "$checked" -ge 4 ] || ok=0
docker rm -f pg_restore_test >/dev/null; rm -f "$W/$key"
if [ $rc -eq 0 ] && [ $ok = 1 ] && [ "$checked" -ge 4 ]; then date -u +%FT%TZ > "$DEST/.restore_verified"; echo "RESTORE VERIFIED ($key)"; else echo "RESTORE NOT VERIFIED (rc=$rc, counts match=$ok)"; fi
