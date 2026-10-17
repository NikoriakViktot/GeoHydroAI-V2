#!/usr/bin/env bash
# Daily mirror of the raw SYNOP archive (synop-fetch: ogimet telegrams, year=/month=/<station>.txt.gz) into the data lake
# in S3, hydro/lake/raw/synop/. Postgres keeps only recent observations (platform synop_observation); history lives here.
# Owner, 2026-10-17. Additive: nothing is deleted in S3.
set -euo pipefail
SRC=/home/ubuntu/synop_lake/data/lake/raw/synop
DST=s3://geo-hydro-ai-data-116344514823-us-east-1-an/hydro/lake/raw/synop
aws s3 sync "$SRC" "$DST" --exclude "fetch.log" --exclude "*.part" --only-show-errors --region us-east-1
n_local=$(find "$SRC" -name "*.txt.gz" | wc -l)
n_s3=$(aws s3 ls "$DST/" --recursive --region us-east-1 | grep -c "\.txt\.gz$" || true)
echo "$(date -u +%FT%TZ) synop_to_s3 local=$n_local s3=$n_s3"
[ "$n_s3" -ge "$n_local" ]
