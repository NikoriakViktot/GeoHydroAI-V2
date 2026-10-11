#services/api/routers/dem.py

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List, Optional

from di import build_online_usecase, online_mode

router = APIRouter(prefix="/dem", tags=["dem"])

class SampleIn(BaseModel):
    dem: str = Field(..., description="Ключ шару як у Terracotta: category/name, напр. dem/fab_dem")
    points: List[List[float]] = Field(..., description="[[lon,lat], ...]")

@router.post("/sample")
def sample_points(body: SampleIn):
    """
    Семпл значень DEM у заданих координатах (читає COG напряму; див. GET /raster/sample).
    """
    if not online_mode():
        return {"error": "ONLINE_SAMPLING=1 required for /dem/sample"}
    uc = build_online_usecase()
    out = []
    for lon, lat in body.points:
        v = uc.sampler.sample(body.dem, lon, lat)  # type: ignore
        out.append({"lon": lon, "lat": lat, "value": v})
    return {"rows": out}
