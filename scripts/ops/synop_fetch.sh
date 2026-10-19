#!/usr/bin/env bash
# Daily SYNOP collection (owner, 2026-10-19: «він же має працювати автоматично»). The 2026-10-16 backfill
# (`synop-fetch`, --since 2000-01) was a one-off container, so nothing fetched new telegrams after it: the S3 mirror
# (synop_to_s3.sh) kept copying the same 1 932 files. This script runs the same fetcher for the previous and the
# current month — src.synop.fetch skips complete past months and always fetches the current one again (and a month
# that ended less than a day ago), politely (one ogimet request at a time, ≥ 5 s apart, backoff on limits) — then
# mirrors the raw archive to S3, where the platform reads the history (SYNOP_LAKE_URI).
#   cron: 20 3 * * * /home/ubuntu/ops/synop_fetch.sh >> /data/backups/synop_fetch.log 2>&1   (replaces the 03:40 mirror line)
set -euo pipefail
DATA=/home/ubuntu/synop_lake/data
IMAGE=geohydroai-hydro-api:latest
SINCE=$(date -u -d "$(date -u +%Y-%m-01) -1 month" +%Y-%m)
echo "$(date -u +%FT%TZ) synop_fetch since $SINCE"
rc=0
docker run --rm --name synop-fetch-daily --entrypoint python -e DATA_DIR=/data -v "$DATA:/data" "$IMAGE" \
  -m src.synop.fetch --since "$SINCE" --data-dir /data || rc=$?
# exit code 2 = ogimet still unavailable after the retries: mirror what was fetched, report the failure
/home/ubuntu/ops/synop_to_s3.sh
latest=$(find "$DATA/lake/raw/synop" -name "*.txt.gz" -newermt "-1 day" | wc -l)
echo "$(date -u +%FT%TZ) synop_fetch done rc=$rc files_updated_24h=$latest"
exit $rc
