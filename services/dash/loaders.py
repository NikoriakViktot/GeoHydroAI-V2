from functools import lru_cache
from pathlib import Path
import pandas as pd
import geopandas as gpd
from utils.db import DuckDBData

def _must_exist(p: Path | str) -> str:
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p}")
    return str(p)

@lru_cache
def get_db_nmad(path: str) -> DuckDBData:
    return DuckDBData(_must_exist(path))

@lru_cache
def read_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(_must_exist(path))

@lru_cache
def read_gpkg_4326(path: str):
    return gpd.read_file(_must_exist(path)).to_crs("EPSG:4326")
