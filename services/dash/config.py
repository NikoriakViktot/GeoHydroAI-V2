# services/dash/config.py
from pathlib import Path
import os
import pandas as pd
import geopandas as gpd
from utils.db import DuckDBData
# якщо не передали, буде /app/data (ми так змонтуємо в compose)
DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data"))

def _must_exist(p: Path) -> Path:
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p}")
    return p

# шляхи
NMAD_PARQUET = _must_exist(DATA_DIR / "NMAD_dem.parquet")
TRACKS_PARQUET_PATH = Path(os.getenv("TRACKS_PARQUET", DATA_DIR / "tracks_3857_1.parquet"))
TRACKS_PARQUET = _must_exist(TRACKS_PARQUET_PATH)
TRACKS_PATH = str(DATA_DIR / "tracks_3857_1.parquet")


BASIN_GPKG = _must_exist(DATA_DIR / "basin_bil_cher_4326.gpkg")
CDF_PARQUET = _must_exist(DATA_DIR / "cdf_precomputed.parquet")
INITIAL_SAMPLE_PARQUET = _must_exist(DATA_DIR / "initial_sample.parquet")
INITIAL_STATS_PARQUET = _must_exist(DATA_DIR / "initial_stats.parquet")
STATS_ALL_PARQUET = _must_exist(DATA_DIR / "stats_all_cached.parquet")

# об'єкти даних / фрейми
# (імпорт DuckDBData зроби там, де він у тебе визначений)
db_NMAD = DuckDBData(str(NMAD_PARQUET))
tracks_parquet_path = str(TRACKS_PARQUET)  # як шлях-рядок для читання/інших місць
tracks_db = DuckDBData(TRACKS_PATH, persistent=True)

gdf_basin = gpd.read_file(BASIN_GPKG).to_crs("EPSG:4326")
cdf_df = pd.read_parquet(CDF_PARQUET)
dff_plot = pd.read_parquet(INITIAL_SAMPLE_PARQUET)
filtered_stats_all_dems = pd.read_parquet(INITIAL_STATS_PARQUET).to_dict("records")
stats_all = pd.read_parquet(STATS_ALL_PARQUET).to_dict("records")

