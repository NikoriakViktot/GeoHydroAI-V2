from dataclasses import dataclass
from typing import Callable, Any, Dict
import config as S
from loaders import get_db_nmad, read_parquet, read_gpkg_4326

@dataclass(frozen=True)
class DataRegistry:
    dfs: Dict[str, Callable[[], Any]]
    dbs: Dict[str, Callable[[], Any]]

    def df(self, key: str):
        return self.dfs[key]()

    def db(self, key: str):
        return self.dbs[key]()

registry = DataRegistry(
    dfs={
        "cdf":             lambda: read_parquet(str(S.CDF_PARQUET)),
        "initial_sample":  lambda: read_parquet(str(S.INITIAL_SAMPLE_PARQUET)),
        "initial_stats":   lambda: read_parquet(str(S.INITIAL_STATS_PARQUET)),
        "stats_all":       lambda: read_parquet(str(S.STATS_ALL_PARQUET)),
        "basin":           lambda: read_gpkg_4326(str(S.BASIN_GPKG)),
    },
    dbs={
        "nmad":            lambda: get_db_nmad(str(S.NMAD_PARQUET)),
    },
)
