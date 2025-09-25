# services/terracotta/init_tc_server.py
from terracotta import  update_settings
from terracotta.server import create_app
from flask import jsonify
from config import DRIVER_PATH, TC_PORT, TC_HOST

update_settings(
    DRIVER_PATH=str(DRIVER_PATH),
    REPROJECTION_METHOD="nearest",   # ок
    RESAMPLING_METHOD="linear"       # ← замість TC_RESAMPLING_METHOD="bilinear"
)

tile_server = create_app()

tile_server.add_url_rule("/health", "health", lambda: jsonify(status="ok"))