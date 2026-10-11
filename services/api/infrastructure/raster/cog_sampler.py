# services/api/infrastructure/raster/cog_sampler.py
"""Point values straight from the COGs that Terracotta serves.

Terracotta renders tiles (colour-mapped, 8-bit PNG) and has no point query, so the browser can only ever
see a colour. This reads the same files (`COG_DATA_DIR`, mounted read-only into this container as well)
and returns the real Float32 value. A layer is addressed exactly like a Terracotta dataset, `category/name`
(e.g. `flood_scenarios/fab_dem_hand_2000_flood_4m`), which maps back to
`<COG_DATA_DIR>/<category>/<name>_cog.tif` — the inverse of `clean_name` in tc_app/update_terracotta_db.py.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import rasterio
from rasterio.warp import transform
from rasterio.windows import Window

LAYER_RE = re.compile(r"^[a-z0-9_]+/[a-z0-9_]+$")
MAX_LAYERS = 4


class BadLayer(ValueError):
    """Key is not `category/name` (lowercase letters, digits, underscore)."""


class LayerNotFound(LookupError):
    """No COG for that key under COG_DATA_DIR."""


@dataclass(frozen=True)
class Sample:
    layer: str
    value: Optional[float]  # None when nodata or outside the raster
    nodata: bool            # True when the pixel is nodata (for flood masks: dry land) or outside
    inside: bool            # False when the point is outside the raster's extent


class CogSampler:
    def __init__(self, root: Optional[str] = None):
        self.root = Path(root or os.getenv("COG_DATA_DIR", "/app/data/cogs")).resolve()

    def path_for(self, layer: str) -> Path:
        if not LAYER_RE.match(layer):
            raise BadLayer(f"layer must look like category/name, got {layer!r}")
        category, name = layer.split("/")
        for fname in (f"{name}_cog.tif", f"{name}.tif"):
            p = (self.root / category / fname).resolve()
            if p.is_relative_to(self.root) and p.is_file():
                return p
        raise LayerNotFound(layer)

    def probe(self, layer: str, lon: float, lat: float) -> Sample:
        path = self.path_for(layer)
        with rasterio.open(path) as ds:
            if ds.crs:
                xs, ys = transform("EPSG:4326", ds.crs, [lon], [lat])
                x, y = xs[0], ys[0]
            else:
                x, y = lon, lat
            row, col = ds.index(x, y)
            if not (0 <= row < ds.height and 0 <= col < ds.width):
                return Sample(layer, None, True, False)
            v = float(ds.read(1, window=Window(col, row, 1, 1))[0, 0])
            is_nodata = math.isnan(v) or (ds.nodata is not None and v == ds.nodata)
            return Sample(layer, None if is_nodata else v, is_nodata, True)

    def sample(self, layer: str, lon: float, lat: float) -> Optional[float]:
        """Drop-in for the old TerracottaSampler (compare_dem, /dem/sample): None on nodata or unknown layer."""
        try:
            return self.probe(layer, lon, lat).value
        except (BadLayer, LayerNotFound):
            return None
