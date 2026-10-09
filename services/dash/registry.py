from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Any, Dict
import config as S
import contextlib
import duckdb
from loaders import get_api_data, get_db_nmad, read_parquet, read_gpkg_4326

@dataclass(frozen=True)
class DataRegistry:
    dfs: Dict[str, Callable[[], Any]]
    dbs: Dict[str, Callable[[], Any]]

    def df(self, key: str):
        try:
            return self.dfs[key]()
        except KeyError as e:
            raise KeyError(f"Unknown df key: {key}") from e

    def db(self, key: str):
        try:
            return self.dbs[key]()
        except KeyError as e:
            raise KeyError(f"Unknown db key: {key}") from e

registry = DataRegistry(
    dfs={
        "cdf":            lambda: read_parquet(str(S.CDF_PARQUET)),
        "initial_sample": lambda: read_parquet(str(S.INITIAL_SAMPLE_PARQUET)),
        "initial_stats":  lambda: read_parquet(str(S.INITIAL_STATS_PARQUET)),
        "stats_all":      lambda: read_parquet(str(S.STATS_ALL_PARQUET)),
        "basin":          lambda: read_gpkg_4326(str(S.BASIN_GPKG)),
    },
    dbs={
        "nmad":           lambda: get_db_nmad(str(S.NMAD_PARQUET)),
        "tracks":         lambda: get_db_nmad(str(S.TRACKS_PARQUET)),
    } if S.ICESAT_BACKEND == "duckdb" else {
        "nmad":           lambda: get_api_data(S.API_URL),
        "tracks":         lambda: get_api_data(S.API_URL),
    },
)

def get_df(key: str):
    return registry.df(key)

def get_db(key: str):
    return registry.db(key)

def icesat_connection():
    """DuckDB-з'єднання для методів DuckDBData(con, ...); для API-бекенда не потрібне."""
    return duckdb.connect() if S.ICESAT_BACKEND == "duckdb" else contextlib.nullcontext()

__all__ = ["DataRegistry", "registry", "get_df", "get_db", "icesat_connection"]
