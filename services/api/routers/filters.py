#services/api/routers/filters.py

from fastapi import APIRouter, Query
from di import build_duck_repo

router = APIRouter(prefix="/filters", tags=["filters"])

@router.get("/lulc")
def get_lulc(dem: str = Query("fab_dem")):
    repo = build_duck_repo()
    return repo.get_unique_lulc_names(dem)  # type: ignore

@router.get("/landform")
def get_landform(dem: str = Query("fab_dem")):
    repo = build_duck_repo()
    return repo.get_unique_landform(dem)  # type: ignore
