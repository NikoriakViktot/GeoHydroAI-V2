# services/api/routers/icesat_v2.py
"""ICESat-2 з PostGIS (icesat_points / icesat_nmad) — дані для Dash."""
from enum import Enum
from functools import lru_cache
from typing import List, Literal, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from di import get_engine
from infrastructure.postgis.icesat_repository import DEMS, IcesatFilter, IcesatRepository

router = APIRouter(prefix="/icesat", tags=["icesat"])

Dem = Enum("Dem", {d: d for d in DEMS}, type=str)


@lru_cache
def _repo() -> IcesatRepository:
    return IcesatRepository(get_engine())


def _range(lo: Optional[float], hi: Optional[float]) -> Optional[list[float]]:
    if lo is None and hi is None:
        return None
    if lo is None or hi is None:
        raise HTTPException(422, "both bounds of a range are required")
    return [lo, hi]


class FilterIn(BaseModel):
    slope_range: Optional[List[float]] = Field(None, min_length=2, max_length=2)
    hand_range: Optional[List[float]] = Field(None, min_length=2, max_length=2)
    lulc: Optional[List[str]] = None
    landform: Optional[List[str]] = None

    def to_filter(self) -> IcesatFilter:
        return IcesatFilter(self.slope_range, self.hand_range, self.lulc or None, self.landform or None)


class StatsIn(FilterIn):
    dems: List[Dem] = Field(default_factory=lambda: list(Dem))


class SampleIn(FilterIn):
    dem: Dem
    n: int = Field(20_000, ge=1, le=200_000)


class GroupedIn(FilterIn):
    by: Literal["slope", "hand", "geomorphon", "lulc"]


@router.get("/tracks")
def tracks(year: int):
    return _repo().tracks(year)


@router.get("/dates")
def dates(track: float, rgt: float, spot: float):
    return _repo().dates(track, rgt, spot)


@router.get("/profile")
def profile(track: float, rgt: float, spot: float, dem: Dem,
            date: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$"),
            hand_min: Optional[float] = None, hand_max: Optional[float] = None):
    return _repo().profile(track, rgt, spot, dem.value, date, _range(hand_min, hand_max))


@router.get("/track-geojson")
def track_geojson(track: float, rgt: float, spot: float, dem: Dem,
                  date: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$"),
                  hand_min: Optional[float] = None, hand_max: Optional[float] = None,
                  step: int = Query(50, ge=1, le=10_000)):
    return _repo().track_geojson(track, rgt, spot, dem.value, date, _range(hand_min, hand_max), step)


@router.get("/filters/lulc")
def filter_lulc(dem: Dem):
    return _repo().lulc_names(dem.value)


@router.get("/filters/landform")
def filter_landform(dem: Dem):
    return _repo().landforms(dem.value)


@router.post("/stats")
def stats(body: StatsIn):
    return _repo().stats(body.to_filter(), [d.value for d in body.dems])


@router.post("/sample")
def sample(body: SampleIn):
    return _repo().sample(body.dem.value, body.to_filter(), body.n)


@router.post("/nmad/grouped")
def nmad_grouped(body: GroupedIn):
    return _repo().nmad_grouped(body.by, body.to_filter())
