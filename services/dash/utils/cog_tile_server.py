# utils/cog_tile_server.py

from flask import send_file
from rasterio.io import MemoryFile
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio import open as rio_open
import numpy as np
from io import BytesIO
import math

tile_server = Blueprint("tile_server", __name__)

COG_PATH = "data/COG/dem/distance_to_stream/aster_dem_distance_to_stream_cog.tif"








TILE_SIZE = 256


def tile_bounds(x, y, z):
    n = 2.0 ** z
    lon_deg_w = x / n * 360.0 - 180.0
    lon_deg_e = (x + 1) / n * 360.0 - 180.0
    lat_rad_n = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat_rad_s = math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n)))
    lat_deg_n = math.degrees(lat_rad_n)
    lat_deg_s = math.degrees(lat_rad_s)
    return (lon_deg_w, lat_deg_s, lon_deg_e, lat_deg_n)


@tile_server.route('/tiles/<int:z>/<int:x>/<int:y>.png')
def serve_tile(z, x, y):
    filepath = "data/COG/dem/distance_to_stream/aster_dem_distance_to_stream_cog.tif"

    try:
        with rio_open(filepath) as src:
            bounds = tile_bounds(x, y, z)
            window = src.window(*bounds)
            data = src.read(1, window=window, out_shape=(TILE_SIZE, TILE_SIZE), resampling=Resampling.bilinear)

            if np.isnan(data).all() or np.ma.is_masked(data) and data.mask.all():
                return send_file(BytesIO(), mimetype="image/png", status=204)

            # Нормалізуй значення
            min_val, max_val = np.percentile(data[~np.isnan(data)], [5, 95])
            data_clipped = np.clip((data - min_val) / (max_val - min_val), 0, 1)
            rgba = (plt.cm.viridis(data_clipped) * 255).astype(np.uint8)

            from PIL import Image
            img = Image.fromarray(rgba)
            buf = BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            return send_file(buf, mimetype="image/png")
    except Exception as e:
        print(f"[Tile ERROR] {z}/{x}/{y}: {e}")
        return send_file(BytesIO(), mimetype="image/png", status=204)
