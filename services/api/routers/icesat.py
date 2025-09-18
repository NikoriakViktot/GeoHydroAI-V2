from fastapi import APIRouter, Body
from pydantic import BaseModel, Field
from typing import List, Optional

from services.api.di import build_online_usecase, online_mode

router = APIRouter(prefix="/icesat", tags=["icesat"])

class IcesatQueryIn(BaseModel):
    aoi_wkt: str = Field(..., description="AOI в WKT (4326)")
    limit: int = Field(10000, ge=1, le=1_000_000)

@router.post("/query")
def query_icesat(body: IcesatQueryIn):
    """
    Повертає точки ICESat-2 усередині AOI (онлайн-режим).
    """
    if not online_mode():
        return {"error": "ONLINE_SAMPLING=1 required for /icesat/query"}
    uc = build_online_usecase()
    # використовуємо репозиторій всередині usecase
    rows = []
    for pid, h, lon, lat in uc.dem_repo.icesat_points_in_aoi(body.aoi_wkt, body.limit):  # type: ignore
        rows.append({"id": pid, "height": h, "lon": lon, "lat": lat})
    return {"rows": rows}

@router.get("/{point_id}")
def icesat_point(point_id: int):
    # Можна додати деталі з PostGIS; для MVP повернемо заглушку
    return {"id": point_id}
