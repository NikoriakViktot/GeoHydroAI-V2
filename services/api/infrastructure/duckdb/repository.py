# infrastructure/duckdb/repository.py
from __future__ import annotations
from typing import Dict, List, Optional
import os, duckdb, pandas as pd

class DuckDBDeltaRepository:
    """
    Репозиторій для офлайнового режиму (Parquet з уже порахованими delta_*, nmad_*).
    За замовчуванням читає файл з env: PARQUET_FILE.
    За бажанням можна розвести на 2 файли (дельти і nmad) — див. __init__.
    """

    def __init__(self, parquet: Optional[str] = None, parquet_nmad: Optional[str] = None):
        self.parquet_deltas = parquet or os.getenv("PARQUET_FILE", "/data/ic2_dem.parquet")
        self.parquet_nmad = parquet_nmad or os.getenv("PARQUET_FILE_NMAD", self.parquet_deltas)

    def _q(self, sql: str, which: str = "deltas") -> pd.DataFrame:
        path = self.parquet_deltas if which == "deltas" else self.parquet_nmad
        con = duckdb.connect()
        try:
            sql = sql.replace("{PARQUET}", path)
            return con.execute(sql).fetchdf()
        finally:
            con.close()

    # ---- приклад агрегації (MAE/RMSE/Bias) за DEM і фільтрами
    def get_filtered_stats(
        self,
        dem: str,
        slope_range: Optional[List[float]] = None,
        hand_range: Optional[List[float]] = None,
        lulc: Optional[List[str]] = None,
        landform: Optional[List[str]] = None,
    ) -> Dict:
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
            filters.append("lulc_name IN (" + ",".join([f"'{x}'" for x in lulc]) + ")")
        if landform:
            filters.append(f"{dem}_landform IN (" + ",".join([f"'{x}'" for x in landform]) + ")")
        where = " AND ".join(filters)
        sql = f"""
            SELECT 
              COUNT(*) as N_points,
              ROUND(AVG(ABS(delta_{dem})), 3) as MAE,
              ROUND(SQRT(AVG(POWER(delta_{dem}, 2))), 3) as RMSE,
              ROUND(AVG(delta_{dem}), 3) as Bias
            FROM '{{PARQUET}}'
            WHERE {where}
        """
        df = self._q(sql, which="deltas")
        return df.iloc[0].to_dict() if not df.empty else {"N_points": 0, "MAE": None, "RMSE": None, "Bias": None}

    # ---- профіль треку/дати у GeoJSON (точки)
    def get_geojson_for_date(
        self,
        track: int,
        rgt: int,
        spot: int,
        dem: str,
        date: str,
        hand_range: Optional[List[float]] = None,
        step: int = 10
    ) -> Dict:
        hand_col = f"{dem}_2000"
        filters = [
            f"track={track}",
            f"rgt={rgt}",
            f"spot={spot}",
            "atl03_cnf = 4",
            "atl08_class = 1",
            f"DATE(time) = '{date}'"
        ]
        if hand_range and len(hand_range) == 2 and all(x is not None for x in hand_range):
            filters.append(f"{hand_col} IS NOT NULL AND {hand_col} BETWEEN {hand_range[0]} AND {hand_range[1]}")
        where = " AND ".join(filters)
        sql = f"""
            SELECT x, y, delta_{dem}, abs_delta_{dem}
            FROM '{{PARQUET}}'
            WHERE {where}
            ORDER BY x
        """
        df = self._q(sql, which="deltas")
        if df.empty:
            return {"type": "FeatureCollection", "features": []}
        df = df.iloc[::max(1, int(step))]
        feats = []
        for _, r in df.iterrows():
            feats.append({
                "type": "Feature",
                "properties": {
                    "delta": float(r[f"delta_{dem}"]),
                    "abs_delta": float(r[f"abs_delta_{dem}"])
                },
                "geometry": {"type": "Point", "coordinates": [float(r["x"]), float(r["y"])]}
            })
        return {"type": "FeatureCollection", "features": feats}

    # ---- фільтри (dropdown)
    def get_unique_lulc_names(self, dem: str):
        sql = f"""
            SELECT DISTINCT lulc_name
            FROM '{{PARQUET}}'
            WHERE delta_{dem} IS NOT NULL AND lulc_name IS NOT NULL
            AND atl03_cnf = 4 AND atl08_class = 1
            ORDER BY lulc_name
        """
        df = self._q(sql, which="deltas")
        return [{"label": x, "value": x} for x in df["lulc_name"].dropna().tolist()]

    def get_unique_landform(self, dem: str):
        col = f"{dem}_landform"
        sql = f"""
            SELECT DISTINCT {col}
            FROM '{{PARQUET}}'
            WHERE {col} IS NOT NULL AND atl03_cnf = 4 AND atl08_class = 1
            ORDER BY {col}
        """
        df = self._q(sql, which="deltas")
        return [{"label": x, "value": x} for x in df[col].dropna().tolist()]
