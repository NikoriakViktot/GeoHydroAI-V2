#!/usr/bin/env bash
# Validate services/nginx/nginx.conf in a throwaway nginx container (same image, certs, docker network as
# geoai-nginx). Fails on errors AND warnings, so deprecated directives do not creep back in.
#   services/nginx/tests/check_config.sh [path/to/nginx.conf]
set -euo pipefail
CONF=$(realpath "${1:-$(dirname "$0")/../nginx.conf}")
IMAGE=${IMAGE:-nginx:stable}
NETWORK=${NETWORK:-geoai_web}
out=$(docker run --rm --network "$NETWORK" -v "$CONF:/etc/nginx/nginx.conf:ro" -v geoai_certs:/etc/letsencrypt:ro \
      --entrypoint nginx "$IMAGE" -t 2>&1) || { echo "$out"; echo "FAIL: nginx -t"; exit 1; }
if grep -E "\[(warn|emerg|alert|crit)\]" <<<"$out"; then
  echo "FAIL: nginx -t reported warnings"; exit 1
fi
echo "OK: $CONF"
