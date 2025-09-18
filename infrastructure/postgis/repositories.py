# infrastructure/postgis/repositories.py
from typing import Iterator, Tuple
import os
from sqlalchemy import create_engine, text

# Використовуємо той самий POSTGRES_URL, що й API
POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql+psycopg://postgres:postgres@postgis:5432/geohydro")

_engine = create_engine(POSTGRES_URL, pool_pre_ping=True, future=True)

class PostgisDemRepository:
    """
    Мінімальний репозиторій для читання точок ICESat-2 з PostGIS всередині AOI (WKT, EPSG:4326).
    """

    def icesat_points_in_wkt(self, wkt: str, limit: int = 200000) -> Iterator[Tuple[int, float, float, float]]:
        sql = text("""
            SELECT id, height, ST_X(geom)::float AS lon, ST_Y(geom)::float AS lat
            FROM icesat_points
            WHERE ST_Intersects(geom, ST_GeomFromText(:wkt,4326))
            LIMIT :lim
        """)
        with _engine.connect() as conn:
            res = conn.execute(sql, {"wkt": wkt, "lim": limit})
            for row in res:
                yield int(row.id), float(row.height), float(row.lon), float(row.lat)
