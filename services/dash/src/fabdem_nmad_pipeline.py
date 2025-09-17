import duckdb
import pandas as pd

# Підключення до DuckDB
con = duckdb.connect(database=':memory:')

# Шлях до parquet-файлу
parquet_path = '/mnt/c/Users/5302/PycharmProjects/GeoHydroAI/data/tracks_3857_1.parquet'


# 1. Глобальні медіани
con.execute(f"""
CREATE OR REPLACE TABLE global_medians AS
SELECT  
  MEDIAN(abs_delta_alos_dem) AS med_alos,
  MEDIAN(abs_delta_aster_dem) AS med_aster,
  MEDIAN(abs_delta_copernicus_dem) AS med_cop,
  MEDIAN(abs_delta_fab_dem) AS med_fab,
  MEDIAN(abs_delta_nasa_dem) AS med_nasa,
  MEDIAN(abs_delta_srtm_dem) AS med_srtm,
  MEDIAN(abs_delta_tan_dem) AS med_tan
FROM '{parquet_path}'
WHERE atl03_cnf = 4 AND atl08_class = 1;
""")

# 2. Об'єднання з медіанами + NMAD
con.execute(f"""
CREATE OR REPLACE TABLE full_with_nmad AS
SELECT 
  t.*,  
  1.4826 * ABS(t.abs_delta_alos_dem - g.med_alos) AS nmad_alos,
  1.4826 * ABS(t.abs_delta_aster_dem - g.med_aster) AS nmad_aster,
  1.4826 * ABS(t.abs_delta_copernicus_dem - g.med_cop) AS nmad_cop,
  1.4826 * ABS(t.abs_delta_fab_dem - g.med_fab) AS nmad_fab,
  1.4826 * ABS(t.abs_delta_nasa_dem - g.med_nasa) AS nmad_nasa,
  1.4826 * ABS(t.abs_delta_srtm_dem - g.med_srtm) AS nmad_srtm,
  1.4826 * ABS(t.abs_delta_tan_dem - g.med_tan) AS nmad_tan
FROM (
    SELECT * FROM '{parquet_path}' 
    WHERE atl03_cnf = 4 AND atl08_class = 1
) AS t
CROSS JOIN global_medians g;
""")


# 5. Збереження результатів
df_best = con.execute("SELECT * FROM full_with_nmad").fetchdf()
df_best.to_parquet("/mnt/c/Users/5302/PycharmProjects/GeoHydroAI/data/NMAD_dem.parquet", index=False)
print("✅ Файл збережено: data/nmad_best_dem.parquet")


# CREATE OR REPLACE TABLE nmad_table AS
# SELECT
#   fab_dem_geomorphon AS geomorphon,
#   fab_dem_slope AS slope_horn,
#   fab_dem_2000 AS hand,
#   fab_dem_twi AS twi,
#   fab_dem_stream AS dist_river,
#   lulc_class AS lulc,
