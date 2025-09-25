#services/api/di.py

import os
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

# Adapters (ти вже маєш їхні реалізації або додай з попереднього меседжа)

from infrastructure.postgis.repositories import PostgisDemRepository
from infrastructure.terr_inf.client import TerracottaSampler
from infrastructure.duckdb.repository import DuckDBDeltaRepository
from core.usecases.compare_dem import CompareDemUseCase

POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql+psycopg://postgres:postgres@postgis:5432/geohydro")
_engine = create_engine(POSTGRES_URL, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False, future=True)

def get_sync_session() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def build_online_usecase() -> CompareDemUseCase:
    repo = PostgisDemRepository()
    sampler = TerracottaSampler(base=os.getenv("TERRACOTTA_URL", "http://terracotta:5000"))
    slope_layer = os.getenv("SLOPE_LAYER", "slope")
    hand_layer = os.getenv("HAND_LAYER", "hand")
    return CompareDemUseCase(repo_points=repo, sampler=sampler, slope_layer=slope_layer, hand_layer=hand_layer)

def build_duck_repo() -> DuckDBDeltaRepository:
    parquet = os.getenv("PARQUET_FILE", "/data/ic2_dem.parquet")
    return DuckDBDeltaRepository(parquet)

def online_mode() -> bool:
    return os.getenv("ONLINE_SAMPLING", "0") == "1"
