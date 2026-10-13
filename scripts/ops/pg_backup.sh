#!/usr/bin/env bash
# Nightly dumps of the host's single PostGIS (geoai-postgis-1): platform, course_db, geohydro. Custom format,
# kept 14 days in /data/backups/pg. Copied to S3 backups/pg/ when the instance role allows it (needs s3:PutObject
# on backups/* — AccessDenied on 2026-10-13, see the server audit). Installed to ~/ops/pg_backup.sh by
# scripts/ops/consolidate_pg.sh; cron: 15 2 * * * /home/ubuntu/ops/pg_backup.sh >> /data/backups/pg_backup.log 2>&1
set -uo pipefail
DEST=/data/backups/pg; TS=$(date -u +%Y%m%dT%H%MZ); BUCKET=geo-hydro-ai-data-116344514823-us-east-1-an; rc=0
dump() {  # db
  local f="$DEST/$1_$TS.dump"
  if docker exec geoai-postgis-1 pg_dump -U postgres -d "$1" -Fc > "$f.part" && [ -s "$f.part" ]; then mv "$f.part" "$f"; echo "ok $f $(du -h "$f" | cut -f1)"
    aws s3 cp "$f" "s3://$BUCKET/backups/pg/$(basename "$f")" --region us-east-1 --only-show-errors 2>/dev/null || echo "s3 copy skipped (no permission?) for $f"
  else rm -f "$f.part"; echo "FAILED $1"; rc=1; fi
}
dump platform
dump course_db
dump geohydro
find "$DEST" -name "*.dump" -mtime +14 -delete
exit $rc
