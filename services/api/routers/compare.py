#services/api/routers/compare.py

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from di import build_online_usecase, build_duck_repo, online_mode

router = APIRouter(prefix="/compare", tags=["compare"])

# ---- Online DEM vs ICESat у AOI (Terracotta семплінг) ----
class CompareAOIIn(BaseModel):
    aoi_wkt: str
    dem_names: List[str] = Field(..., description="['Copernicus','FABDEM',...]")
    limit: int = 200000

@router.post("/stats")
def compare_stats(body: CompareAOIIn):
    """
    ONLINE (якщо ONLINE_SAMPLING=1): семплінг DEM ↔ ICESat-2 у AOI, повертає метрики.
    OFFLINE (ONLINE_SAMPLING=0): читає вже пораховані delta_* з Parquet і повертає метрики.
    """
    if online_mode():
        uc = build_online_usecase()
        res = uc.execute(body.aoi_wkt, body.dem_names, body.limit)
        return {"mode": "online", **res}
    else:
        repo = build_duck_repo()
        # Для простоти повернемо по одному DEM (можна розширити до списку)
        metrics: Dict[str, Any] = {}
        for dem in body.dem_names:
            m = repo.get_filtered_stats(dem=dem)  # type: ignore
            metrics[dem] = m
        return {"mode": "offline", "metrics": metrics}

# ---- Профіль треку/дати (DuckDB parquet) ----
class ProfileIn(BaseModel):
    dem: str
    track: int
    rgt: int
    spot: int
    date: str
    hand_range: Optional[List[float]] = None
    step: int = 10

@router.post("/profile")
def compare_profile(body: ProfileIn):
    """
    OFFLINE (Parquet/DuckDB): повертає GeoJSON точок з delta/abs_delta для вибраного профілю.
    """
    repo = build_duck_repo()
    gj = repo.get_geojson_for_date(
        track=body.track, rgt=body.rgt, spot=body.spot,
        dem=body.dem, date=body.date, hand_range=body.hand_range, step=body.step  # type: ignore
    )
    return gj
