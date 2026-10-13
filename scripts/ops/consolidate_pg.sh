#!/usr/bin/env bash
# Three Postgres -> one PostGIS (geoai-postgis-1). Server audit 2026-10-13, memory step 4.
# Preconditions: GeoHydroAI-V2 #14 merged and deployed (postgis on the official image + python_course network),
#                geoai-django #24 merged and `git pull`ed in ~/geoai-django (compose without platform-db).
# Run as ubuntu on the server:  bash ~/geoai/scripts/ops/consolidate_pg.sh
# Idempotent up to the cutover; ~5 min; platform-api down ~2 min, python-course ~1 min. Old volumes are kept.
set -euo pipefail
PG=geoai-postgis-1; TS=$(date -u +%Y%m%dT%H%MZ); DEST=/data/backups/pg; LOG=$DEST/consolidate_$TS.log
exec > >(tee -a "$LOG") 2>&1
echo "== $TS preconditions"
docker inspect -f '{{.Config.Image}}' $PG | grep -q '^postgis/postgis:16-3.5$' || { echo "postgis is not on the official image yet (deploy GeoHydroAI-V2 #14 first)"; exit 1; }
docker network inspect python_curse_default -f '{{range .Containers}}{{.Name}} {{end}}' | grep -q "$PG" || { echo "$PG is not on python_curse_default yet"; exit 1; }
grep -q 'geoai-postgis-1:5432/platform' ~/geoai-django/deploy/server/docker-compose.platform.yml || { echo "~/geoai-django is not on the compose without platform-db (git pull after merging geoai-django #24)"; exit 1; }

echo "== fresh backups"
docker exec platform-db pg_dump -U platform -d platform -Fc > $DEST/platform_pre_$TS.dump
docker exec python_curse-db-1 pg_dump -U course_user -d course_db -Fc > $DEST/course_db_pre_$TS.dump
ls -la $DEST/*_pre_$TS.dump | awk '{print $5, $9}'

echo "== drop unused extensions in geohydro (0 vector columns, 0 foreign servers on 2026-10-13)"
docker exec $PG psql -U postgres -d geohydro -v ON_ERROR_STOP=1 -c "DROP EXTENSION IF EXISTS parquet_fdw; DROP EXTENSION IF EXISTS ogr_fdw; DROP EXTENSION IF EXISTS vector;"

echo "== roles (password hashes copied container-to-container, never printed)"
docker exec platform-db pg_dumpall -U platform --roles-only 2>/dev/null | grep -vE '^(CREATE|ALTER) ROLE postgres\b' | docker exec -i $PG psql -U postgres -q -v ON_ERROR_STOP=0 >/dev/null
docker exec python_curse-db-1 pg_dumpall -U course_user --roles-only 2>/dev/null | grep -vE '^(CREATE|ALTER) ROLE postgres\b' | docker exec -i $PG psql -U postgres -q -v ON_ERROR_STOP=0 >/dev/null
docker exec $PG psql -U postgres -tAc "select string_agg(rolname,' ') from pg_roles where rolname in ('platform','icesat2','course_user')"

echo "== databases"
docker exec $PG psql -U postgres -tAc "select 1 from pg_database where datname='platform'" | grep -q 1 || docker exec $PG psql -U postgres -c "CREATE DATABASE platform OWNER platform"
docker exec $PG psql -U postgres -tAc "select 1 from pg_database where datname='course_db'" | grep -q 1 || docker exec $PG psql -U postgres -c "CREATE DATABASE course_db OWNER course_user"
docker exec $PG psql -U postgres -d platform -c "CREATE EXTENSION IF NOT EXISTS postgis" >/dev/null

echo "== cutover platform (API down from here)"
cd ~/geoai-django
docker compose -f deploy/server/docker-compose.platform.yml --env-file ~/geoai-django-secrets/platform.env stop platform-worker platform-api
docker exec platform-db pg_dump -U platform -d platform -Fc | docker exec -i $PG pg_restore -U postgres -d platform --no-owner --role=platform --no-privileges --exit-on-error
BEFORE=$(docker exec platform-db psql -U platform -d platform -tAc "select count(*) from pg_tables where schemaname='public'")
AFTER=$(docker exec $PG psql -U postgres -d platform -tAc "select count(*) from pg_tables where schemaname='public'")
echo "platform tables: before=$BEFORE after=$AFTER"; [ "$BEFORE" = "$AFTER" ]
docker exec $PG psql -U postgres -d platform -c "ALTER SCHEMA public OWNER TO platform; GRANT ALL ON SCHEMA public TO platform;" >/dev/null
docker compose -f deploy/server/docker-compose.platform.yml --env-file ~/geoai-django-secrets/platform.env up -d --remove-orphans platform-redis platform-api platform-worker
for i in $(seq 1 30); do s=$(docker inspect -f '{{.State.Health.Status}}' platform-api 2>/dev/null || echo starting); [ "$s" = healthy ] && break; sleep 5; done; echo "platform-api: $s"; [ "$s" = healthy ]

echo "== cutover Python_Curse"
cd ~/Python_Curse
python3 - <<'PY'
import re
p='docker-compose.yml'; s=open(p).read()
if '\n  db:\n' in s:
    s=s[:s.index('  db:\n')]+s[s.index('  redis:\n'):]                 # drop the db service, keep redis
s=re.sub(r'    depends_on:\n      db:\n        condition: service_healthy\n      redis:\n        condition: service_started\n',
         '    depends_on:\n      redis:\n        condition: service_started\n', s)
s=re.sub(r'  certbot:\n(?:    .*\n)+?(?=volumes:)', '', s)             # renewed by geoai-certbot-renewer (geoai_certs volume)
s=s.replace('  postgres_data:\n','').replace('  certbot_www:\n','').replace('  certbot_conf:\n','')
if 'POSTGRES_HOST' not in s:
    s=s.replace('      - DJANGO_SETTINGS_MODULE=config.settings.local\n','      - DJANGO_SETTINGS_MODULE=config.settings.local\n      - POSTGRES_HOST=geoai-postgis-1\n')
open(p,'w').write(s)
PY
docker compose config --services | tr '\n' ' '; echo
docker compose stop web worker beat
docker exec python_curse-db-1 pg_dump -U course_user -d course_db -Fc | docker exec -i $PG pg_restore -U postgres -d course_db --no-owner --role=course_user --no-privileges --exit-on-error
docker exec $PG psql -U postgres -d course_db -c "ALTER SCHEMA public OWNER TO course_user; GRANT ALL ON SCHEMA public TO course_user;" >/dev/null
docker compose up -d --remove-orphans web worker beat
sleep 15; curl -s -o /dev/null -w "python-course: %{http_code}\n" https://python-course-viktor-nikoriak.org/

echo "== stop the old databases (volumes kept: geohydroai-platform_platform_pgdata, python_curse_postgres_data)"
docker stop platform-db python_curse-db-1 python_curse-certbot-1 2>/dev/null || true
docker rm platform-db python_curse-db-1 python_curse-certbot-1 2>/dev/null || true
git -C ~/Python_Curse add docker-compose.yml && git -C ~/Python_Curse commit -qm "deploy: database and certbot move to the host's shared PostGIS / certbot-renewer (server audit 2026-10-13)" || true

echo "== nightly backup now dumps the three databases from $PG"
install -m 755 ~/geoai/scripts/ops/pg_backup.sh ~/ops/pg_backup.sh
echo "== done"; free -m | sed -n 2,3p; docker ps --format '{{.Names}}' | grep -E 'db|postgis'
