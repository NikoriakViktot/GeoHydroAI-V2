import duckdb
import pandas as pd

# Підключення до DuckDB (у пам’яті або збережений файл)
con = duckdb.connect(database=':memory:')

# Шлях до повного parquet з усіма полями
parquet_path = '/mnt/c/Users/5302/PycharmProjects/GeoHydroAI/data/tracks_3857_1.parquet'

# 1. Обчислюємо медіани
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
WHERE
  atl03_cnf = 4 AND atl08_class = 1;

""")

# 2. Обчислюємо NMAD і зберігаємо лише ці стовпці
con.execute(f"""
CREATE OR REPLACE TABLE nmad_only AS
SELECT 
  1.4826 * ABS(t.abs_delta_alos_dem - g.med_alos) AS nmad_alos,
  1.4826 * ABS(t.abs_delta_aster_dem - g.med_aster) AS nmad_aster,
  1.4826 * ABS(t.abs_delta_copernicus_dem - g.med_cop) AS nmad_cop,
  1.4826 * ABS(t.abs_delta_fab_dem - g.med_fab) AS nmad_fab,
  1.4826 * ABS(t.abs_delta_nasa_dem - g.med_nasa) AS nmad_nasa,
  1.4826 * ABS(t.abs_delta_srtm_dem - g.med_srtm) AS nmad_srtm,
  1.4826 * ABS(t.abs_delta_tan_dem - g.med_tan) AS nmad_tan

FROM '{parquet_path}' t
CROSS JOIN global_medians g
WHERE
  atl03_cnf = 4 AND atl08_class = 1;
""")

# 3. Зберігаємо тільки NMAD у файл
df_nmad = con.execute("SELECT * FROM nmad_only").fetchdf()
df_nmad.to_parquet("/mnt/c/Users/5302/PycharmProjects/GeoHydroAI/data/nmad_all.parquet", index=False)

print("✅ Збережено  NMAD")
