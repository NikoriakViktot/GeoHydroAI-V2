# services/terracotta/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

# Підтягнути змінні з .env (локально)
load_dotenv()

GEOTIFF_DIR = os.getenv("COG_DATA_DIR", "data/cogs")
DRIVER_PATH = Path(os.getenv("TC_DB", "tc_db/terracotta.sqlite"))
PROJECTION = "EPSG:3857"
TILE_PATH_TEMPLATE = "/tiles/{category}/{name}/{z}/{x}/{y}.png"

TC_PORT = int(os.getenv("TC_PORT", "5000"))
TC_HOST = os.getenv("TC_HOST", "0.0.0.0")
TC_URL  = f"http://{TC_HOST}:{TC_PORT}"


DEM_KEYS = ["alos", "aster", "copernicus", "fab", "nasa", "srtm", "tan"]
LULC_YEARS = list(range(2018, 2026))