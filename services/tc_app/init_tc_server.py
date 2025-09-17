# services/terracotta/init_tc_server.py
from terracotta import  update_settings
from terracotta.server import create_app
from config import DRIVER_PATH, TC_PORT, TC_HOST

update_settings(DRIVER_PATH=DRIVER_PATH, REPROJECTION_METHOD="nearest")
tile_server = create_app()
