# services/api/infrastructure/postgis/icesat_repository.py
"""
Запити до icesat_points / icesat_nmad (див. services/postgis/init/02_icesat.sql).

Назви колонок підставляються в SQL лише з білого списку DEMS,
усі значення — через bind-параметри.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Optional, Sequence

from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine

DEMS: tuple[str, ...] = (
    "alos_dem", "aster_dem", "copernicus_dem", "fab_dem",
    "nasa_dem", "srtm_dem", "tan_dem",
)
NMAD_COLS: tuple[str, ...] = (
    "nmad_alos", "nmad_aster", "nmad_cop", "nmad_fab",
    "nmad_nasa", "nmad_srtm", "nmad_tan",
)
# icesat_nmad уже містить лише atl03_cnf=4 AND atl08_class=1; для icesat_points фільтр явний
GOOD_PHOTONS = "atl03_cnf = 4 AND atl08_class = 1"


def _check_dem(dem: str) -> str:
    if dem not in DEMS:
        raise ValueError(f"unknown dem: {dem}")
    return dem


@dataclass
class IcesatFilter:
    slope_range: Optional[Sequence[float]] = None
    hand_range: Optional[Sequence[float]] = None
    lulc: Optional[list[str]] = None
    landform: Optional[list[str]] = None


@dataclass
class _Where:
    """Накопичує умови WHERE і bind-параметри."""
    parts: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    expanding: set[str] = field(default_factory=set)

    def add(self, sql: str, **params: Any) -> None:
        self.parts.append(sql)
        self.params.update(params)

    def add_in(self, col: str, name: str, values: list[str]) -> None:
        self.parts.append(f"{col} IN :{name}")
        self.params[name] = list(values)
        self.expanding.add(name)

    def sql(self) -> str:
        return " AND ".join(self.parts) if self.parts else "TRUE"

    def text(self, sql: str):
        stmt = text(sql)
        if self.expanding:
            stmt = stmt.bindparams(*(bindparam(n, expanding=True) for n in self.expanding))
        return stmt


def _filter_conds(w: _Where, f: IcesatFilter, dem: str, suffix: str = "", with_lulc: bool = True) -> list[str]:
    """Умови фільтра для колонок конкретного DEM; повертає список SQL-умов."""
    conds: list[str] = []
    if f.slope_range:
        conds.append(f"{dem}_slope BETWEEN :slo{suffix} AND :shi{suffix}")
        w.params.update({f"slo{suffix}": f.slope_range[0], f"shi{suffix}": f.slope_range[1]})
    if f.hand_range:
        conds.append(f"{dem}_2000 BETWEEN :hlo{suffix} AND :hhi{suffix}")
        w.params.update({f"hlo{suffix}": f.hand_range[0], f"hhi{suffix}": f.hand_range[1]})
    if with_lulc and f.lulc:
        conds.append("lulc_name IN :lulc")
        w.params["lulc"] = list(f.lulc)
        w.expanding.add("lulc")
    if f.landform:
        conds.append(f"{dem}_landform IN :lf{suffix}")
        w.params[f"lf{suffix}"] = list(f.landform)
        w.expanding.add(f"lf{suffix}")
    return conds


def _day_bounds(date: str) -> tuple[dt.datetime, dt.datetime]:
    d = dt.date.fromisoformat(date)
    start = dt.datetime.combine(d, dt.time.min)
    return start, start + dt.timedelta(days=1)


class IcesatRepository:
    def __init__(self, engine: Engine):
        self.engine = engine

    def _rows(self, stmt, params: dict) -> tuple[list[str], list[tuple]]:
        with self.engine.connect() as conn:
            res = conn.execute(stmt, params)
            return list(res.keys()), [tuple(r) for r in res]

    # --- треки / дати (icesat_points) ---
    def tracks(self, year: int) -> list[dict]:
        sql = text(f"""
            SELECT DISTINCT track, rgt, spot FROM icesat_points
            WHERE year = :year AND {GOOD_PHOTONS}
            ORDER BY track, rgt, spot
        """)
        cols, rows = self._rows(sql, {"year": year})
        return [dict(zip(cols, r)) for r in rows]

    def dates(self, track: float, rgt: float, spot: float) -> list[str]:
        sql = text(f"""
            SELECT DISTINCT time::date AS d FROM icesat_points
            WHERE track = :track AND rgt = :rgt AND spot = :spot AND {GOOD_PHOTONS}
            ORDER BY d
        """)
        _, rows = self._rows(sql, {"track": track, "rgt": rgt, "spot": spot})
        return [r[0].isoformat() for r in rows]

    def _track_where(self, track, rgt, spot, date) -> _Where:
        start, end = _day_bounds(date)
        w = _Where()
        w.add("track = :track AND rgt = :rgt AND spot = :spot",
              track=track, rgt=rgt, spot=spot)
        w.add("time >= :t0 AND time < :t1", t0=start, t1=end)
        w.add(GOOD_PHOTONS)
        return w

    def profile(self, track, rgt, spot, dem: str, date: str,
                hand_range: Optional[Sequence[float]] = None) -> dict:
        dem = _check_dem(dem)
        w = self._track_where(track, rgt, spot, date)
        w.add(f"delta_{dem} IS NOT NULL AND h_{dem} IS NOT NULL")
        if hand_range:
            w.add(f"{dem}_2000 BETWEEN :hlo AND :hhi", hlo=hand_range[0], hhi=hand_range[1])
        sql = f"""
            SELECT track, rgt, spot, time, x, y, orthometric_height,
                   h_{dem}, delta_{dem}, abs_delta_{dem}, {dem}_2000
            FROM icesat_points WHERE {w.sql()} ORDER BY x
        """
        cols, rows = self._rows(w.text(sql), w.params)
        return {"columns": cols, "rows": rows}

    def track_geojson(self, track, rgt, spot, dem: str, date: str,
                      hand_range: Optional[Sequence[float]] = None, step: int = 50) -> dict:
        """Точки треку для карти; delta = COALESCE(delta_dem, orthometric_height - h_dem), кожна step-та точка по x."""
        dem = _check_dem(dem)
        w = self._track_where(track, rgt, spot, date)
        w.add(f"orthometric_height IS NOT NULL AND h_{dem} IS NOT NULL")
        if hand_range:
            w.add(f"{dem}_2000 BETWEEN :hlo AND :hhi", hlo=hand_range[0], hhi=hand_range[1])
        w.params["step"] = max(int(step), 1)
        sql = f"""
            WITH base AS (
                SELECT x, y, orthometric_height,
                       COALESCE(delta_{dem}, orthometric_height - h_{dem}) AS delta
                FROM icesat_points WHERE {w.sql()}
            ), numbered AS (
                SELECT x, y, orthometric_height, delta,
                       ROW_NUMBER() OVER (ORDER BY x) AS rn
                FROM base WHERE delta IS NOT NULL
            )
            SELECT x, y, orthometric_height, delta FROM numbered
            WHERE :step <= 1 OR rn % :step = 1
            ORDER BY rn
        """
        _, rows = self._rows(w.text(sql), w.params)
        return {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [x, y]},
                    "properties": {"delta": delta, "orthometric_height": oh},
                }
                for x, y, oh, delta in rows
            ],
        }

    # --- фільтри (icesat_nmad; дані статичні між імпортами -> кеш у процесі) ---
    @lru_cache(maxsize=32)
    def lulc_names(self, dem: str) -> list[str]:
        dem = _check_dem(dem)
        sql = text(f"""
            SELECT DISTINCT lulc_name FROM icesat_nmad
            WHERE delta_{dem} IS NOT NULL AND lulc_name IS NOT NULL
            ORDER BY lulc_name
        """)
        _, rows = self._rows(sql, {})
        return [r[0] for r in rows]

    @lru_cache(maxsize=32)
    def landforms(self, dem: str) -> list[str]:
        dem = _check_dem(dem)
        sql = text(f"""
            SELECT DISTINCT {dem}_landform AS v FROM icesat_nmad
            WHERE {dem}_landform IS NOT NULL
            ORDER BY v
        """)
        _, rows = self._rows(sql, {})
        return [r[0] for r in rows]

    # --- статистика (icesat_nmad) ---
    def stats(self, f: IcesatFilter, dems: Sequence[str] = DEMS) -> list[dict]:
        """N_points/MAE/RMSE/Bias для кожного DEM одним проходом по таблиці.
        Фільтри slope/HAND/landform застосовуються до колонок відповідного DEM."""
        w = _Where()
        if f.lulc:
            w.add_in("lulc_name", "lulc", f.lulc)
        selects, any_dem = [], []
        for i, dem in enumerate(map(_check_dem, dems)):
            cond = " AND ".join([f"delta_{dem} IS NOT NULL", *_filter_conds(w, f, dem, suffix=str(i), with_lulc=False)])
            any_dem.append(f"({cond})")
            d = f"delta_{dem}"
            selects.append(f"""
                COUNT(*) FILTER (WHERE {cond}) AS "{dem}.N_points",
                ROUND(AVG(ABS({d})) FILTER (WHERE {cond})::numeric, 2)::float AS "{dem}.MAE",
                ROUND(SQRT(AVG({d} * {d}) FILTER (WHERE {cond}))::numeric, 2)::float AS "{dem}.RMSE",
                ROUND(AVG({d}) FILTER (WHERE {cond})::numeric, 2)::float AS "{dem}.Bias"
            """)
        w.add("(" + " OR ".join(any_dem) + ")")
        sql = f"SELECT {', '.join(selects)} FROM icesat_nmad WHERE {w.sql()}"
        cols, rows = self._rows(w.text(sql), w.params)
        row = dict(zip(cols, rows[0]))
        out = []
        for dem in dems:
            rec = {"DEM": dem}
            for k in ("N_points", "MAE", "RMSE", "Bias"):
                rec[k] = row[f"{dem}.{k}"]
            out.append(rec)
        return out

    def sample(self, dem: str, f: IcesatFilter, n: int = 20_000) -> list[float]:
        """Випадкова вибірка delta_{dem} з урахуванням фільтрів (для гістограми/boxplot)."""
        dem = _check_dem(dem)
        w = _Where()
        w.add(f"delta_{dem} IS NOT NULL")
        w.parts += _filter_conds(w, f, dem)
        w.params["n"] = int(n)
        sql = f"SELECT delta_{dem} FROM icesat_nmad WHERE {w.sql()} ORDER BY random() LIMIT :n"
        _, rows = self._rows(w.text(sql), w.params)
        return [r[0] for r in rows]

    # --- медіани NMAD за класами (фільтри завжди по fab_dem, як у дашборді) ---
    _GROUPS = {
        "slope": ("slope_class", """CASE
                WHEN fab_dem_slope < 5 THEN '0–5°'
                WHEN fab_dem_slope < 10 THEN '5–10°'
                WHEN fab_dem_slope < 15 THEN '10–15°'
                WHEN fab_dem_slope < 20 THEN '15–20°'
                WHEN fab_dem_slope < 25 THEN '20–25°'
                WHEN fab_dem_slope < 30 THEN '25–30°'
                ELSE '>30°' END"""),
        "hand": ("hand_class", """CASE
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
                ELSE '>10 м' END"""),
    }

    def nmad_grouped(self, by: str, f: IcesatFilter) -> dict:
        w = _Where()
        w.parts += _filter_conds(w, f, "fab_dem")
        medians = ", ".join(
            f"percentile_cont(0.5) WITHIN GROUP (ORDER BY {c}) AS {c}" for c in NMAD_COLS
        )
        if by in self._GROUPS:
            alias, expr = self._GROUPS[by]
            sql = f"""
                SELECT * FROM (
                    SELECT {expr} AS {alias}, {medians}
                    FROM icesat_nmad WHERE {w.sql()}
                    GROUP BY 1
                ) g ORDER BY {alias} COLLATE "C"
            """
        elif by == "geomorphon":
            w.add("fab_dem_geomorphon IS NOT NULL")
            sql = f"""
                SELECT fab_dem_geomorphon AS landform, {medians}
                FROM icesat_nmad WHERE {w.sql()}
                GROUP BY fab_dem_geomorphon ORDER BY fab_dem_geomorphon
            """
        elif by == "lulc":
            sql = f"""
                SELECT lulc_class, lulc_name, {medians}
                FROM icesat_nmad WHERE {w.sql()}
                GROUP BY lulc_class, lulc_name ORDER BY lulc_class
            """
        else:
            raise ValueError(f"unknown group: {by}")
        cols, rows = self._rows(w.text(sql), w.params)
        return {"columns": cols, "rows": rows}
