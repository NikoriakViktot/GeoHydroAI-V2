# services/dash/config.py
from pathlib import Path
import os


DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data"))

NMAD_PARQUET            = DATA_DIR / "NMAD_dem.parquet"
TRACKS_PARQUET          = Path(os.getenv("TRACKS_PARQUET", str(DATA_DIR / "tracks_3857_1.parquet")))
BASIN_GPKG              = DATA_DIR / "basin_bil_cher_4326.gpkg"
CDF_PARQUET             = DATA_DIR / "cdf_precomputed.parquet"
INITIAL_SAMPLE_PARQUET  = DATA_DIR / "initial_sample.parquet"
INITIAL_STATS_PARQUET   = DATA_DIR / "initial_stats.parquet"
STATS_ALL_PARQUET       = DATA_DIR / "stats_all_cached.parquet"

# джерело даних ICESat: "api" (PostGIS через API) або "duckdb" (parquet напряму, для відкату)
ICESAT_BACKEND          = os.getenv("ICESAT_BACKEND", "api").strip().lower()
API_URL                 = os.getenv("API_URL", "http://api:8000")
