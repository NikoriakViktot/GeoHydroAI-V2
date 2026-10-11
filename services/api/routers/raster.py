# services/api/routers/raster.py
"""GET /api/raster/sample — the number under the cursor.

The platform (Django, `GET /api/v1/raster/sample`) proxies this for the SPA so the map can show
"depth 1.8 m" for the HAND flood mask instead of a colour. Reads the COG directly (see CogSampler).
"""
from fastapi import APIRouter, HTTPException, Query

from infrastructure.raster.cog_sampler import MAX_LAYERS, BadLayer, CogSampler, LayerNotFound

router = APIRouter(prefix="/raster", tags=["raster"])


@router.get("/sample")
def sample(
    layers: str = Query(..., description=f"Comma-separated Terracotta keys `category/name`, at most {MAX_LAYERS}, "
                                         "e.g. `flood_scenarios/fab_dem_hand_2000_flood_4m,hand_2000/fab_dem_hand_2000`"),
    lon: float = Query(..., ge=-180, le=180, description="EPSG:4326"),
    lat: float = Query(..., ge=-90, le=90, description="EPSG:4326"),
):
    """Raster value at a point, one entry per layer, in the raster's own units (flood masks and HAND: metres).

    `nodata: true` means the pixel holds the raster's nodata value — for flood masks that is dry land, not a
    depth of 0 — or the point lies outside the raster (`inside: false`). Values are never interpolated.
    """
    keys = [k.strip() for k in layers.split(",") if k.strip()]
    if not keys or len(keys) > MAX_LAYERS:
        raise HTTPException(422, detail=f"layers: give 1 to {MAX_LAYERS} comma-separated keys")
    sampler = CogSampler()
    out = []
    for key in keys:
        try:
            s = sampler.probe(key, lon, lat)
        except BadLayer as e:
            raise HTTPException(422, detail=str(e))
        except LayerNotFound:
            raise HTTPException(404, detail=f"layer not found: {key}")
        out.append({"layer": s.layer, "value": s.value, "nodata": s.nodata, "inside": s.inside})
    return {"lon": lon, "lat": lat, "crs": "EPSG:4326", "values": out}
