#!/bin/sh
set -e
echo ">>> Terracotta | indexing COGs from $COG_DATA_DIR into $TC_DB"
python /app/tc_app/update_terracotta_db.py

echo ">>> Terracotta | starting server on ${TC_HOST}:${TC_PORT}"
exec gunicorn -w 2 -k gevent -b ${TC_HOST}:${TC_PORT} init_tc_server:tile_server
