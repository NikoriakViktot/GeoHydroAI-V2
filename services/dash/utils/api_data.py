# utils/api_data.py
"""
Клієнт ICESat-ендпоінтів API (/icesat/*) з тими самими методами, що й DuckDBData,
щоб callbacks не залежали від джерела даних (див. ICESAT_BACKEND у config.py).
Аргумент `con` у методах лишено для сумісності з DuckDBData — тут він ігнорується.
"""
from __future__ import annotations

import logging

import pandas as pd
import requests

logger = logging.getLogger(__name__)

NMAD_COLS = ["nmad_alos", "nmad_aster", "nmad_cop", "nmad_fab", "nmad_nasa", "nmad_srtm", "nmad_tan"]
EMPTY_FC = {"type": "FeatureCollection", "features": []}


def _hand_params(hand_range) -> dict:
    if hand_range and len(hand_range) == 2 and all(x is not None for x in hand_range):
        return {"hand_min": hand_range[0], "hand_max": hand_range[1]}
    return {}


def _filter_body(slope_range=None, hand_range=None, lulc=None, landform=None) -> dict:
    return {
        "slope_range": list(slope_range) if slope_range else None,
        "hand_range": list(hand_range) if hand_range else None,
        "lulc": list(lulc) if lulc else None,
        "landform": list(landform) if landform else None,
    }


def _frame(payload: dict) -> pd.DataFrame:
    return pd.DataFrame(payload["rows"], columns=payload["columns"])


def _with_best(df: pd.DataFrame) -> pd.DataFrame:
    if not df.empty:
        df["best_dem"] = df[NMAD_COLS].idxmin(axis=1)
        df["best_nmad"] = df[NMAD_COLS].min(axis=1)
    return df


class ApiData:
    def __init__(self, base_url: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/") + "/icesat"
        self.timeout = timeout
        self.session = requests.Session()

    def _get(self, path: str, **params):
        r = self.session.get(self.base_url + path, params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, body: dict):
        r = self.session.post(self.base_url + path, json=body, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    # --- dropdowns / фільтри ---
    def get_track_dropdown_options(self, year):
        try:
            rows = self._get("/tracks", year=year)
        except requests.RequestException as e:
            logger.error("get_track_dropdown_options failed: %s", e)
            return []
        df = pd.DataFrame(rows, columns=["track", "rgt", "spot"])
        return [
            {"label": f"Track {row.track} / RGT {row.rgt} / Spot {row.spot}",
             "value": f"{row.track}_{row.rgt}_{row.spot}"}
            for _, row in df.iterrows()
        ]

    def get_date_dropdown_options(self, track, rgt, spot):
        try:
            dates = self._get("/dates", track=track, rgt=rgt, spot=spot)
        except requests.RequestException as e:
            logger.error("get_date_dropdown_options failed: %s", e)
            return []
        return [{"label": d, "value": d} for d in dates]

    def get_unique_lulc_names(self, dem):
        if not dem:
            logger.warning("get_unique_lulc_names: DEM is None -> returning empty list")
            return []
        return [{"label": x, "value": x} for x in self._get("/filters/lulc", dem=dem)]

    def get_unique_landform(self, dem):
        return [{"label": x, "value": x} for x in self._get("/filters/landform", dem=dem)]

    # --- профіль / карта треку ---
    def get_profile(self, track, rgt, spot, dem, date, hand_range=None):
        try:
            df = _frame(self._get("/profile", track=track, rgt=rgt, spot=spot, dem=dem, date=date,
                                  **_hand_params(hand_range)))
        except requests.RequestException as e:
            logger.error("get_profile failed: %s", e)
            return pd.DataFrame()
        if "time" in df:
            df["time"] = pd.to_datetime(df["time"])
        return df

    def get_geojson_for_date(self, track, rgt, spot, dem, date, hand_range=None, step=50):
        try:
            return self._get("/track-geojson", track=track, rgt=rgt, spot=spot, dem=dem, date=date,
                             step=step, **_hand_params(hand_range))
        except requests.RequestException as e:
            logger.error("get_geojson_for_date failed: %s", e)
            return EMPTY_FC

    def get_dem_stats(self, df, dem_key):
        delta_col = f"delta_{dem_key}"
        if delta_col not in df:
            return None
        delta = df[delta_col].dropna()
        if delta.empty:
            return None
        return {"mean": delta.mean(), "min": delta.min(), "max": delta.max(), "count": len(delta)}

    # --- головна вкладка ---
    def get_filtered_sample(self, con, dem, slope_range=None, hand_range=None, lulc=None, landform=None,
                            sample_n=10000):
        body = {"dem": dem, "n": sample_n, **_filter_body(slope_range, hand_range, lulc, landform)}
        return pd.DataFrame({f"delta_{dem}": self._post("/sample", body)})

    def get_filtered_stats_all(self, con, dems, slope_range=None, hand_range=None, lulc=None, landform=None):
        body = {"dems": list(dems), **_filter_body(slope_range, hand_range, lulc, landform)}
        return self._post("/stats", body)

    # --- Best DEM (медіани NMAD за класами) ---
    def _nmad_grouped(self, by, slope_range, hand_range, lulc, landform):
        body = {"by": by, **_filter_body(slope_range, hand_range, lulc, landform)}
        return _with_best(_frame(self._post("/nmad/grouped", body)))

    def get_nmad_grouped_by_slope(self, con, nmad_cols=None, slope_range=None, hand_range=None, lulc=None,
                                  landform=None):
        return self._nmad_grouped("slope", slope_range, hand_range, lulc, landform)

    def get_nmad_grouped_by_hand(self, con, nmad_cols=None, slope_range=None, hand_range=None, lulc=None,
                                 landform=None):
        return self._nmad_grouped("hand", slope_range, hand_range, lulc, landform)

    def get_nmad_grouped_by_geomorphon(self, con, nmad_cols=None, slope_range=None, hand_range=None, lulc=None,
                                       landform=None):
        return self._nmad_grouped("geomorphon", slope_range, hand_range, lulc, landform)

    def get_nmad_grouped_by_lulc(self, con, nmad_cols=None, slope_range=None, hand_range=None, lulc=None,
                                 landform=None):
        return self._nmad_grouped("lulc", slope_range, hand_range, lulc, landform)
