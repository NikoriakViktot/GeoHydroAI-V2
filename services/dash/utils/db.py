# db.py
import duckdb
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DuckDBData:
    def __init__(self, parquet_file, persistent=False):
        self.parquet_file = parquet_file
        self.persistent = persistent
        self.con = duckdb.connect() if persistent else None

    def query(self, sql):
        try:
            if self.persistent:
                return self.con.execute(sql).fetchdf()
            else:
                with duckdb.connect() as con:
                    return con.execute(sql).fetchdf()
        except Exception as e:
            print(f"DuckDB query error: {e}")
            return pd.DataFrame()

    def close(self):
        if self.persistent and self.con:
            self.con.close()

    # --- 1. Стартові ключі профілю ---
    def get_default_profile_keys(self, dem="alos_dem"):
        sql = f"""
            SELECT track, rgt, spot, '{dem}' as dem, DATE(time) as date_only
            FROM '{self.parquet_file}'
            WHERE atl03_cnf = 4 AND atl08_class = 1 AND delta_{dem} IS NOT NULL
            LIMIT 1
        """
        df = self.query(sql)
        if not df.empty:
            return tuple(df.iloc[0])
        return (None, None, None, None, None)

    # --- 2. Всі унікальні треки для року ---
    def get_unique_tracks(self, year):
        sql = (
            f"SELECT DISTINCT track, rgt, spot FROM '{self.parquet_file}' "
            f"WHERE year = {year} AND atl03_cnf = 4 AND atl08_class = 1 "
            "ORDER BY track, rgt, spot"
        )
        return self.query(sql)

    # --- 3. Всі дати для треку ---
    def get_unique_dates(self, track, rgt, spot):
        sql = (
            f"SELECT DISTINCT DATE(time) as date_only FROM '{self.parquet_file}' "
            f"WHERE track={track} AND rgt={rgt} AND spot={spot} "
            "AND atl03_cnf = 4 AND atl08_class = 1 "
            "ORDER BY date_only"
        )
        return self.query(sql)

    # --- 4. Отримати профіль треку для дати/DEM ---
    def get_profile(self, track, rgt, spot, dem, date, hand_range=None):
        hand_col = f"{dem}_2000"
        sql = (
            f"SELECT * FROM '{self.parquet_file}' "
            f"WHERE track={track} AND rgt={rgt} AND spot={spot} "
            f"AND DATE(time) = '{date}' "
            f"AND delta_{dem} IS NOT NULL AND h_{dem} IS NOT NULL "
            "AND atl03_cnf = 4 AND atl08_class = 1"
        )
        if hand_range and len(hand_range) == 2 and all(x is not None for x in hand_range):
            sql += f" AND {hand_col} IS NOT NULL AND {hand_col} BETWEEN {hand_range[0]} AND {hand_range[1]}"
        sql += " ORDER BY x"
        return self.query(sql)

    # --- 5. Статистика для профілю (mean/min/max/count) ---
    def get_dem_stats(self, df, dem_key):
        delta_col = f"delta_{dem_key}"
        if delta_col not in df:
            return None
        delta = df[delta_col].dropna()
        if delta.empty:
            return None
        return {
            "mean": delta.mean(),
            "min": delta.min(),
            "max": delta.max(),
            "count": len(delta)
        }


    # --- 6. Time Series для треку (по днях) ---
    def get_time_series(self, track, rgt, spot, dem):
        sql = (
            f"SELECT DATE(time) as date_only, AVG(abs_delta_{dem}) as mean_abs_error "
            f"FROM '{self.parquet_file}' "
            f"WHERE track={track} AND rgt={rgt} AND spot={spot} "
            "AND atl03_cnf = 4 AND atl08_class = 1 "
            f"AND abs_delta_{dem} IS NOT NULL "
            "GROUP BY date_only ORDER BY date_only"
        )
        return self.query(sql)

    # --- 7. GEOJSON для карти треку ---
    def get_geojson_for_date(self, track, rgt, spot, dem, date, hand_range=None, step=10):
        hand_col = f"{dem}_2000"
        sql = (
            f"SELECT x, y, delta_{dem}, abs_delta_{dem} FROM '{self.parquet_file}' "
            f"WHERE track={track} AND rgt={rgt} AND spot={spot} "
            f"AND DATE(time) = '{date}' "
            "AND atl03_cnf = 4 AND atl08_class = 1"
        )
        if hand_range and len(hand_range) == 2 and all(x is not None for x in hand_range):
            sql += f" AND {hand_col} IS NOT NULL AND {hand_col} BETWEEN {hand_range[0]} AND {hand_range[1]}"
        df = self.query(sql)
        if df.empty:
            return {"type": "FeatureCollection", "features": []}
        df = df.iloc[::step]
        features = [
            {
                "type": "Feature",
                "properties": {
                    "delta": float(r.get(f"delta_{dem}")),
                    "abs_delta": float(r.get(f"abs_delta_{dem}"))
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(r.x), float(r.y)]
                }
            }
            for _, r in df.iterrows()
        ]
        return {"type": "FeatureCollection", "features": features}

    # --- 8. LULC/landform для фільтрів ---


    def get_unique_lulc_names(self, dem: str | None):
        # якщо DEM невідомий — нічого не фільтруємо і нічого не ламаємо
        if not dem:
            logger.warning("get_unique_lulc_names: DEM is None -> returning empty list")
            return []
        try:
            sql = f"""
                SELECT DISTINCT lulc_name
                FROM '{self.path}'
                WHERE delta_{dem} IS NOT NULL AND lulc_name IS NOT NULL
            """
            df = duckdb.query(sql).to_df()
            if "lulc_name" not in df:
                return []
            return [{"label": x, "value": x} for x in df["lulc_name"].dropna().tolist()]
        except Exception as e:
            logger.error("get_unique_lulc_names failed: %s", e)
            return []
    def get_unique_landform(self, dem):
        landform_col = f"{dem}_landform"
        sql = f"""
            SELECT DISTINCT {landform_col}
            FROM '{self.parquet_file}'
            WHERE {landform_col} IS NOT NULL
            AND atl03_cnf = 4 AND atl08_class = 1
            ORDER BY {landform_col}
        """
        df = self.query(sql)
        return [{"label": x, "value": x} for x in df[landform_col].dropna().tolist()]


    # --- 9. Dropdowns ---
    def get_track_dropdown_options(self, year):
        df = self.get_unique_tracks(year)
        return [
            {"label": f"Track {row.track} / RGT {row.rgt} / Spot {row.spot}",
             "value": f"{row.track}_{row.rgt}_{row.spot}"}
            for _, row in df.iterrows()
        ]

    def get_date_dropdown_options(self, track, rgt, spot):
        df = self.get_unique_dates(track, rgt, spot)
        return [
            {
                "label": pd.to_datetime(row.date_only).strftime("%Y-%m-%d"),
                "value": pd.to_datetime(row.date_only).strftime("%Y-%m-%d")
            }
            for _, row in df.iterrows()
        ]

    def get_filtered_sample(self, con, dem, slope_range=None, hand_range=None, lulc=None, landform=None,
                            sample_n=10000):
        filters = [
            f"delta_{dem} IS NOT NULL",
            "atl03_cnf = 4",
            "atl08_class = 1"
        ]
        if slope_range:
            filters.append(f"{dem}_slope BETWEEN {slope_range[0]} AND {slope_range[1]}")
        if hand_range:
            filters.append(f"{dem}_2000 BETWEEN {hand_range[0]} AND {hand_range[1]}")
        if lulc:
            lulc_str = ",".join([f"'{x}'" for x in lulc])
            filters.append(f"lulc_name IN ({lulc_str})")
        if landform:
            landform_str = ",".join([f"'{x}'" for x in landform])
            filters.append(f"{dem}_landform IN ({landform_str})")
        where = " AND ".join(filters)
        sql = f"""
              SELECT * FROM '{self.parquet_file}'
              WHERE {where}
              USING SAMPLE {sample_n} ROWS
          """
        return con.execute(sql).fetchdf()

    def get_filtered_data(self, con, dem, slope_range=None, hand_range=None, lulc=None, landform=None, cols=None):
        filters = [
            f"delta_{dem} IS NOT NULL",
            "atl03_cnf = 4",
            "atl08_class = 1"
        ]
        if slope_range:
            filters.append(f"{dem}_slope BETWEEN {slope_range[0]} AND {slope_range[1]}")
        if hand_range:
            filters.append(f"{dem}_2000 BETWEEN {hand_range[0]} AND {hand_range[1]}")
        if lulc:
            lulc_str = ",".join([f"'{x}'" for x in lulc])
            filters.append(f"lulc_name IN ({lulc_str})")
        if landform:
            landform_str = ",".join([f"'{x}'" for x in landform])
            filters.append(f"{dem}_landform IN ({landform_str})")
        where = " AND ".join(filters)
        if cols:
            col_str = ", ".join(cols)
        else:
            col_str = "*"
        sql = f"SELECT {col_str} FROM '{self.parquet_file}' WHERE {where}"
        return con.execute(sql).fetchdf()



    def get_filtered_stats(self, con, dem, slope_range=None, hand_range=None, lulc=None, landform=None):
        filters = [
            f"delta_{dem} IS NOT NULL",
            "atl03_cnf = 4",
            "atl08_class = 1"
        ]
        if slope_range:
            filters.append(f"{dem}_slope BETWEEN {slope_range[0]} AND {slope_range[1]}")
        if hand_range:
            filters.append(f"{dem}_2000 BETWEEN {hand_range[0]} AND {hand_range[1]}")
        if lulc:
            lulc_str = ",".join([f"'{x}'" for x in lulc])
            filters.append(f"lulc_name IN ({lulc_str})")
        if landform:
            landform_str = ",".join([f"'{x}'" for x in landform])
            filters.append(f"{dem}_landform IN ({landform_str})")
        where = " AND ".join(filters)
        sql = f"""
            SELECT 
                COUNT(*) as N_points,
                ROUND(AVG(ABS(delta_{dem})),2) as MAE,
                ROUND(SQRT(AVG(POWER(delta_{dem}, 2))), 2) as RMSE,
                ROUND(AVG(delta_{dem}), 2) as Bias
            FROM '{self.parquet_file}'
            WHERE {where}
        """
        df = con.execute(sql).fetchdf()
        return df.iloc[0].to_dict()

    # utils/db.py

    def get_dem_stats_sql(self, con, dem_key, hand_range=None):
        filters = [
            f"delta_{dem_key} IS NOT NULL",
            "atl03_cnf = 4",
            "atl08_class = 1"
        ]
        if hand_range:
            filters.append(f"{dem_key}_2000 BETWEEN {hand_range[0]} AND {hand_range[1]}")
        filter_str = " AND ".join(filters)
        sql = f"""
                   SELECT 
            COUNT(delta_{dem_key}) as N_points,
            ROUND(AVG(ABS(delta_{dem_key})), 3) as MAE,
            ROUND(SQRT(AVG(POWER(delta_{dem_key}, 2))), 3) as RMSE,
            ROUND(AVG(delta_{dem_key}), 3) as Bias
            FROM '{self.parquet_file}'
            WHERE {filter_str}
        """
        df = con.execute(sql).fetchdf()
        if not df.empty and df.iloc[0].N_points:
            res = df.iloc[0].to_dict()
            res["DEM"] = dem_key
            return res
        return None

    def load_nmad_values(self):
        dem_list = [
            "alos", "aster", "cop", "fab",
            "nasa", "srtm", "tan"
        ]
        columns = ", ".join([f"nmad_{dem}" for dem in dem_list])
        query = f"""
            SELECT {columns}
            FROM '{self.parquet_file}'
            WHERE 
                nmad_alos IS NOT NULL AND nmad_alos < 40 AND
                nmad_aster IS NOT NULL AND nmad_aster < 40 AND
                nmad_cop IS NOT NULL AND nmad_cop < 40 AND
                nmad_fab IS NOT NULL AND nmad_fab < 40 AND
                nmad_nasa IS NOT NULL AND nmad_nasa < 40 AND
                nmad_srtm IS NOT NULL AND nmad_srtm < 40 AND
                nmad_tan IS NOT NULL AND nmad_tan < 40
        """
        return self.query(query)

    def get_cdf_from_duckdb(self, thresholds=np.arange(0, 41, 1)):

        con = duckdb.connect()
        dem_list = ["alos", "aster", "cop", "fab", "nasa", "srtm", "tan"]

        cdf_data = []

        for dem in dem_list:
            dem_col = f"nmad_{dem}"
            union_sql = "\nUNION ALL\n".join([
                f"""
                SELECT {t} AS threshold,
                       COUNT(*) FILTER (WHERE {dem_col} <= {t}) * 1.0 / COUNT(*) AS cdf
                FROM '{self.parquet_file}'
                WHERE {dem_col} IS NOT NULL
                """ for t in thresholds
            ])
            df = con.execute(union_sql).fetchdf()
            df["DEM"] = dem.upper()
            cdf_data.append(df)

        con.close()
        return pd.concat(cdf_data, ignore_index=True)

    def clean_df_for_table(self, df):
        # Видалити складні/сервісні колонки для таблиць Dash
        if df.empty:
            return df
        drop_cols = []
        for col in df.columns:
            if df[col].dtype == "object":
                s = df[col].dropna()
                if not s.empty:
                    first_val = s.iloc[0]
                    if isinstance(first_val, (dict, list, tuple, bytes)) or hasattr(first_val, "__array__"):
                        drop_cols.append(col)
                else:
                    drop_cols.append(col)
        if "geometry_bbox" in df.columns:
            drop_cols.append("geometry_bbox")
        return df.drop(columns=list(set(drop_cols)), errors="ignore")

    def get_dem_stats(self, df, dem_key):
        # Повертає mean, min, max, count по вибраному DEM
        delta_col = f"delta_{dem_key}"
        if delta_col not in df:
            return None
        delta = df[delta_col].dropna()
        if delta.empty:
            return None
        return {
            "mean": delta.mean(),
            "min": delta.min(),
            "max": delta.max(),
            "count": len(delta)
        }

    def get_nmad_grouped_by_slope(self, con, nmad_cols=None, slope_range=None, hand_range=None, lulc=None,
                                  landform=None):
        if nmad_cols is None:
            nmad_cols = [
                "nmad_alos", "nmad_aster", "nmad_cop",
                "nmad_fab", "nmad_nasa", "nmad_srtm", "nmad_tan"
            ]
        filters = ["atl03_cnf = 4", "atl08_class = 1"]
        if slope_range:
            filters.append(f"fab_dem_slope BETWEEN {slope_range[0]} AND {slope_range[1]}")
        if hand_range:
            filters.append(f"fab_dem_2000 BETWEEN {hand_range[0]} AND {hand_range[1]}")
        if lulc:
            lulc_str = ",".join([f"'{x}'" for x in lulc])
            filters.append(f"lulc_name IN ({lulc_str})")
        if landform:
            landform_str = ",".join([f"'{x}'" for x in landform])
            filters.append(f"fab_dem_landform IN ({landform_str})")
        where = " AND ".join(filters)
        select_cols = ", ".join([f"MEDIAN({c}) as {c}" for c in nmad_cols])
        sql = f"""
            SELECT slope_class, {select_cols}
            FROM (
                SELECT *,
                    CASE
                        WHEN fab_dem_slope < 5 THEN '0–5°'
                        WHEN fab_dem_slope < 10 THEN '5–10°'
                        WHEN fab_dem_slope < 15 THEN '10–15°'
                        WHEN fab_dem_slope < 20 THEN '15–20°'
                        WHEN fab_dem_slope < 25 THEN '20–25°'
                        WHEN fab_dem_slope < 30 THEN '25–30°'
                        ELSE '>30°'
                    END AS slope_class
                FROM '{self.parquet_file}'
                WHERE {where}
            )
            GROUP BY slope_class
            ORDER BY slope_class
        """
        df = con.execute(sql).fetchdf()
        if not df.empty:
            df["best_dem"] = df[nmad_cols].idxmin(axis=1)
            df["best_nmad"] = df[nmad_cols].min(axis=1)
        return df

    def get_nmad_grouped_by_geomorphon(self, con, nmad_cols=None, slope_range=None, hand_range=None, lulc=None,
                                       landform=None):
        if nmad_cols is None:
            nmad_cols = [
                "nmad_alos", "nmad_aster", "nmad_cop",
                "nmad_fab", "nmad_nasa", "nmad_srtm", "nmad_tan"
            ]
        filters = ["atl03_cnf = 4", "atl08_class = 1"]
        if slope_range:
            filters.append(f"fab_dem_slope BETWEEN {slope_range[0]} AND {slope_range[1]}")
        if hand_range:
            filters.append(f"fab_dem_2000 BETWEEN {hand_range[0]} AND {hand_range[1]}")
        if lulc:
            lulc_str = ",".join([f"'{x}'" for x in lulc])
            filters.append(f"lulc_name IN ({lulc_str})")
        if landform:
            landform_str = ",".join([f"'{x}'" for x in landform])
            filters.append(f"fab_dem_landform IN ({landform_str})")
        where = " AND ".join(filters)
        select_cols = ", ".join([f"MEDIAN({c}) as {c}" for c in nmad_cols])
        sql = f"""
            SELECT fab_dem_geomorphon as landform, {select_cols}
            FROM '{self.parquet_file}'
            WHERE {where} AND fab_dem_geomorphon IS NOT NULL
            GROUP BY fab_dem_geomorphon
            ORDER BY fab_dem_geomorphon
        """
        df = con.execute(sql).fetchdf()
        if not df.empty:
            df["best_dem"] = df[nmad_cols].idxmin(axis=1)
            df["best_nmad"] = df[nmad_cols].min(axis=1)
        return df

    def get_nmad_grouped_by_hand(self, con, nmad_cols=None, slope_range=None, hand_range=None, lulc=None,
                                 landform=None):
        if nmad_cols is None:
            nmad_cols = [
                "nmad_alos", "nmad_aster", "nmad_cop",
                "nmad_fab", "nmad_nasa", "nmad_srtm", "nmad_tan"
            ]
        filters = ["atl03_cnf = 4", "atl08_class = 1"]
        if slope_range:
            filters.append(f"fab_dem_slope BETWEEN {slope_range[0]} AND {slope_range[1]}")
        if hand_range:
            filters.append(f"fab_dem_2000 BETWEEN {hand_range[0]} AND {hand_range[1]}")
        if lulc:
            lulc_str = ",".join([f"'{x}'" for x in lulc])
            filters.append(f"lulc_name IN ({lulc_str})")
        if landform:
            landform_str = ",".join([f"'{x}'" for x in landform])
            filters.append(f"fab_dem_landform IN ({landform_str})")
        where = " AND ".join(filters)
        # Розбиваємо на класи
        sql = f"""
            SELECT hand_class, {', '.join([f"MEDIAN({c}) as {c}" for c in nmad_cols])}
            FROM (
                SELECT *,
                    CASE
                        WHEN fab_dem_2000 < 1 THEN '0–1 м'
                        WHEN fab_dem_2000 < 2 THEN '1–2 м'
                        WHEN fab_dem_2000 < 3 THEN '2–3 м'
                        WHEN fab_dem_2000 < 4 THEN '3–4 м'
                        WHEN fab_dem_2000 < 5 THEN '4–5 м'
                        WHEN fab_dem_2000 < 6 THEN '5–6 м'
                        WHEN fab_dem_2000 < 7 THEN '6–7 м'
                        WHEN fab_dem_2000 < 8 THEN '7–8 м'
                        WHEN fab_dem_2000 < 9 THEN '8–9 м'
                        WHEN fab_dem_2000 < 10 THEN '9–10 м'
                        ELSE '>10 м'
                    END AS hand_class
                FROM '{self.parquet_file}'
                WHERE {where}
            )
            GROUP BY hand_class
            ORDER BY hand_class
        """
        df = con.execute(sql).fetchdf()
        if not df.empty:
            df["best_dem"] = df[nmad_cols].idxmin(axis=1)
            df["best_nmad"] = df[nmad_cols].min(axis=1)
        return df



    def get_nmad_grouped_by_lulc(self, con, nmad_cols=None, slope_range=None, hand_range=None, lulc=None,
                                 landform=None):
        if nmad_cols is None:
            nmad_cols = [
                "nmad_alos", "nmad_aster", "nmad_cop",
                "nmad_fab", "nmad_nasa", "nmad_srtm", "nmad_tan"
            ]
        filters = ["atl03_cnf = 4", "atl08_class = 1"]
        if slope_range:
            filters.append(f"fab_dem_slope BETWEEN {slope_range[0]} AND {slope_range[1]}")
        if hand_range:
            filters.append(f"fab_dem_2000 BETWEEN {hand_range[0]} AND {hand_range[1]}")
        if lulc:
            lulc_str = ",".join([f"'{x}'" for x in lulc])
            filters.append(f"lulc_name IN ({lulc_str})")
        if landform:
            landform_str = ",".join([f"'{x}'" for x in landform])
            filters.append(f"fab_dem_landform IN ({landform_str})")
        where = " AND ".join(filters)
        select_cols = ", ".join([f"MEDIAN({c}) as {c}" for c in nmad_cols])
        sql = f"""
            SELECT lulc_class, lulc_name, {select_cols}
            FROM '{self.parquet_file}'
            WHERE {where}
            GROUP BY lulc_class, lulc_name
            ORDER BY lulc_class
        """
        df = con.execute(sql).fetchdf()
        if not df.empty:
            nmad_columns = nmad_cols
            df["best_dem"] = df[nmad_columns].idxmin(axis=1)
            df["best_nmad"] = df[nmad_columns].min(axis=1)
        return df

#
#
# class DuckDBData:
#     def __init__(self, parquet_file, persistent=False):
#         """
#         Initializes the data access object for a given Parquet file.
#         :param parquet_file: Path to the Parquet file.
#         :param persistent: If True, keeps the database connection open.
#         """
#         self.parquet_file = parquet_file
#         self.persistent = persistent
#         self.con = duckdb.connect() if persistent else None



    def get_track_options_for_year(self, year: int) -> list[dict]:
        """
        Gets unique tracks for a given year to populate a dropdown.
        """
        sql = f"""
            SELECT DISTINCT track, rgt, spot
            FROM '{self.parquet_file}'
            WHERE year = {year}
              AND atl03_cnf = 4 AND atl08_class = 1
            ORDER BY track, rgt, spot
        """
        df = self.query(sql)
        if df.empty:
            return []
        return [
            {"label": f"Track {row.track} / RGT {row.rgt} / Spot {row.spot}",
             "value": f"{row.track}_{row.rgt}_{row.spot}"}
            for _, row in df.iterrows()
        ]

    def get_date_options_for_track(self, track: float, rgt: float, spot: float) -> list[dict]:
        """
        Gets unique dates for a specific track to populate a dropdown.
        """
        sql = f"""
            SELECT DISTINCT DATE(time) as date_only
            FROM '{self.parquet_file}'
            WHERE track={track} AND rgt={rgt} AND spot={spot}
              AND atl03_cnf = 4 AND atl08_class = 1
            ORDER BY date_only
        """
        df = self.query(sql)
        if df.empty:
            return []
        return [{
            "label": pd.to_datetime(row.date_only).strftime("%Y-%m-%d"),
            "value": pd.to_datetime(row.date_only).strftime("%Y-%m-%d")
        } for _, row in df.iterrows()]

    def get_track_profile(self, track: float, rgt: float, spot: float, dem: str, date: str,
                          hand_range: list | None = None) -> pd.DataFrame:
        """
        Fetches the full profile data for a specific track, date, and DEM.
        """
        hand_col = f"{dem}_2000"  # Assuming this column naming convention
        sql = f"""
            SELECT *
            FROM '{self.parquet_file}'
            WHERE track={track} AND rgt={rgt} AND spot={spot}
              AND DATE(time) = '{date}'
              AND delta_{dem} IS NOT NULL AND h_{dem} IS NOT NULL
              AND atl03_cnf = 4 AND atl08_class = 1
        """
        if hand_range and len(hand_range) == 2 and all(x is not None for x in hand_range):
            sql += f" AND {hand_col} IS NOT NULL AND {hand_col} BETWEEN {hand_range[0]} AND {hand_range[1]}"
        sql += " ORDER BY x"
        return self.query(sql)



    def get_track_geojson(self, track: float, rgt: float, spot: float, dem: str, date: str, step: int = 10) -> dict:
        """
        Fetches track data and formats it as a GeoJSON-compliant dictionary.
        This avoids serialization errors in Dash callbacks.
        """
        # We only need a few columns for the GeoJSON representation
        sql = f"""
            SELECT x, y, delta_{dem}
            FROM '{self.parquet_file}'
            WHERE track={track} AND rgt={rgt} AND spot={spot}
              AND DATE(time) = '{date}'
              AND delta_{dem} IS NOT NULL
              AND atl03_cnf = 4 AND atl08_class = 1
            ORDER BY x
        """
        df = self.query(sql)
        if df.empty:
            return {"type": "FeatureCollection", "features": []}

        # Subsample the data to avoid overloading the browser
        df_sampled = df.iloc[::step]

        features = []
        for _, row in df_sampled.iterrows():
            delta_val = row.get(f"delta_{dem}")
            feature = {
                "type": "Feature",
                "properties": {
                    "delta": float(delta_val) if pd.notna(delta_val) else None,
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(row.x), float(row.y)]
                }
            }
            features.append(feature)

        return {"type": "FeatureCollection", "features": features}

