#!/usr/bin/env bash
# Nightly dumps of the host's single PostGIS (geoai-postgis-1): platform, course_db, geohydro. Custom format.
# Scheme (owner, 2026-10-17): every dump goes to S3 backups/pg/ (long-term; the bucket's lifecycle rule ages it out),
# locally only the last KEEP_LOCAL dumps per database stay, and a local dump is removed ONLY after its S3 copy was
# verified (head-object size == local size). The instance role may put/get/list backups/pg/* but not delete.
# geohydro (national layers, ~1.2 GB per dump) changes only when the layers are reloaded: it is dumped only when its
# write fingerprint differs from the last dumped one, or when the last dump is older than GEOHYDRO_MAX_AGE_DAYS (so a
# copy always exists inside the bucket's 365-day expiry). platform and course_db are dumped every night. Owner, 2026-10-17.
# Source: GeoHydroAI-V2 scripts/ops/pg_backup.sh.
set -uo pipefail
DEST=/data/backups/pg; TS=$(date -u +%Y%m%dT%H%MZ); BUCKET=geo-hydro-ai-data-116344514823-us-east-1-an; REGION=us-east-1
KEEP_LOCAL=${KEEP_LOCAL:-3}; VERIFIED="$DEST/.s3_verified"; rc=0
touch "$VERIFIED"

s3_size() {  # key -> size in bytes, empty when missing
  aws s3api head-object --bucket "$BUCKET" --key "$1" --region "$REGION" --query ContentLength --output text 2>/dev/null
}
upload() {  # local file -> verified copy in S3 (records the file name in $VERIFIED)
  local f="$1" key="backups/pg/$(basename "$1")" local_size remote
  local_size=$(stat -c %s "$f")
  if grep -qx "$(basename "$f")" "$VERIFIED"; then return 0; fi
  remote=$(s3_size "$key")
  if [ "$remote" != "$local_size" ]; then
    aws s3 cp "$f" "s3://$BUCKET/$key" --region "$REGION" --only-show-errors || { echo "s3 copy FAILED for $f"; return 1; }
    remote=$(s3_size "$key")
  fi
  if [ "$remote" = "$local_size" ]; then echo "s3 verified $key ($local_size bytes)"; basename "$f" >> "$VERIFIED"; return 0; fi
  echo "s3 size mismatch for $key: local $local_size, remote ${remote:-none}"; return 1
}
fingerprint() {  # db -> hash of the cumulative writes per table + the table list (changes on any DML or DDL)
  docker exec geoai-postgis-1 psql -U postgres -d "$1" -Atc \
    "select md5(coalesce(string_agg(schemaname||'.'||relname||':'||(n_tup_ins+n_tup_upd+n_tup_del), ',' order by schemaname, relname),'')) from pg_stat_user_tables" 2>/dev/null
}
GEOHYDRO_MAX_AGE_DAYS=${GEOHYDRO_MAX_AGE_DAYS:-180}; FP_FILE="$DEST/.geohydro_fingerprint"
dump_if_changed() {  # db: skip when unchanged since the last dump that reached S3
  local fp last newest
  fp=$(fingerprint "$1"); last=$(cat "$FP_FILE" 2>/dev/null)
  newest=$(ls -1t "$DEST"/$1_*.dump 2>/dev/null | head -1)
  if [ -n "$fp" ] && [ "$fp" = "$last" ] && [ -n "$newest" ] && grep -qx "$(basename "$newest")" "$VERIFIED" \
     && [ -n "$(find "$newest" -mtime -"$GEOHYDRO_MAX_AGE_DAYS")" ]; then
    echo "skip $1: unchanged since $(basename "$newest")"; return 0
  fi
  dump "$1" && [ -n "$fp" ] && echo "$fp" > "$FP_FILE"
}
dump() {  # db
  local f="$DEST/$1_$TS.dump"
  if docker exec geoai-postgis-1 pg_dump -U postgres -d "$1" -Fc > "$f.part" && [ -s "$f.part" ]; then
    mv "$f.part" "$f"; echo "ok $f $(du -h "$f" | cut -f1)"
    upload "$f" || { rc=1; return 1; }
  else rm -f "$f.part"; echo "FAILED $1"; rc=1; return 1; fi
}
dump platform
dump course_db
dump_if_changed geohydro

# retry the S3 copy of older local dumps that were never verified (e.g. the days before the role had the permission)
for f in "$DEST"/*.dump; do [ -e "$f" ] && upload "$f" || true; done

# local retention: per database keep the newest KEEP_LOCAL dumps; older ones go only when their S3 copy is verified
# AND only once a restore from S3 was proven (ops/restore_test.sh writes .restore_verified): owner's rule 2026-10-17
[ -e "$DEST/.restore_verified" ] || { echo "retention skipped: no verified restore yet ($DEST/.restore_verified)"; exit $rc; }
for db in platform course_db geohydro; do
  ls -1t "$DEST"/${db}_*.dump 2>/dev/null | tail -n +$((KEEP_LOCAL + 1)) | while read -r f; do
    if grep -qx "$(basename "$f")" "$VERIFIED"; then rm -f "$f" && echo "local removed (S3 verified) $(basename "$f")"
    else echo "kept locally (no verified S3 copy yet) $(basename "$f")"; fi
  done
done
exit $rc
