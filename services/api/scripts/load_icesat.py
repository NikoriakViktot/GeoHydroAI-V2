# services/api/scripts/load_icesat.py
"""
Імпорт ICESat-2 з parquet у PostGIS (icesat_points, icesat_nmad).

    docker compose exec api python -m scripts.load_icesat [--only points|nmad]

Схема таблиць: services/postgis/init/02_icesat.sql (має бути застосована заздалегідь).
Скрипт ідемпотентний: TRUNCATE -> COPY -> індекси -> ANALYZE -> звірка кількості рядків.
"""
import argparse
import io
import logging
import os
import sys
import time

import psycopg
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("load_icesat")

PARQUET_DIR = os.getenv("PARQUET_DIR", "/app/data/parquet")
BATCH_SIZE = 100_000

DEMS = ["alos", "aster", "copernicus", "fab", "nasa", "srtm", "tan"]
NMAD = ["alos", "aster", "cop", "fab", "nasa", "srtm", "tan"]

BASE_COLS = [
    "track", "rgt", "spot", "cycle", "time", "year", "atl03_cnf", "atl08_class",
    "height", "orthometric_height", "x", "y", "lulc_class", "lulc_name",
]
DEM_COLS = [
    c
    for d in DEMS
    for c in (f"h_{d}_dem", f"delta_{d}_dem", f"abs_delta_{d}_dem", f"{d}_dem_slope",
              f"{d}_dem_2000", f"{d}_dem_landform", f"{d}_dem_geomorphon")
]

TABLES = {
    "points": {
        "table": "icesat_points",
        "file": "tracks_3857_1.parquet",
        "cols": BASE_COLS + DEM_COLS,
        "indexes": [
            "CREATE INDEX ix_icesat_points_track ON icesat_points (track, rgt, spot, time)",
            "CREATE INDEX ix_icesat_points_year ON icesat_points (year)",
            "CREATE INDEX ix_icesat_points_geom ON icesat_points USING GIST (geom)",
        ],
        "cluster": "ix_icesat_points_track",
    },
    "nmad": {
        "table": "icesat_nmad",
        "file": "NMAD_dem.parquet",
        "cols": BASE_COLS + DEM_COLS + [f"nmad_{n}" for n in NMAD],
        "indexes": [
            "CREATE INDEX ix_icesat_nmad_lulc ON icesat_nmad (lulc_name)",
            "CREATE INDEX ix_icesat_nmad_year ON icesat_nmad (year)",
        ],
    },
}


def _conninfo() -> str:
    url = os.getenv("POSTGRES_URL", "postgresql+psycopg://postgres:postgres@postgis:5432/geohydro")
    # SQLAlchemy-URL -> libpq-URL
    return url.replace("postgresql+psycopg://", "postgresql://", 1)


def _drop_indexes(cur, table: str) -> None:
    cur.execute(
        "SELECT indexname FROM pg_indexes WHERE tablename = %s AND indexname NOT LIKE %s",
        (table, "%_pkey"),
    )
    for (name,) in cur.fetchall():
        cur.execute(f'DROP INDEX IF EXISTS "{name}"')


def load(conn: psycopg.Connection, spec: dict) -> None:
    table, cols = spec["table"], spec["cols"]
    path = os.path.join(PARQUET_DIR, spec["file"])
    pf = pq.ParquetFile(path)
    expected = pf.metadata.num_rows
    log.info("%s <- %s (%d rows, %d cols)", table, path, expected, len(cols))

    t0 = time.monotonic()
    with conn.cursor() as cur:
        _drop_indexes(cur, table)
        cur.execute(f"TRUNCATE {table} RESTART IDENTITY")
        conn.commit()

        copy_sql = f"COPY {table} ({', '.join(cols)}) FROM STDIN WITH (FORMAT csv)"
        write_opts = pacsv.WriteOptions(include_header=False)
        done = 0
        with cur.copy(copy_sql) as copy:
            for batch in pf.iter_batches(columns=cols, batch_size=BATCH_SIZE):
                buf = io.BytesIO()
                pacsv.write_csv(batch, buf, write_opts)
                copy.write(buf.getvalue())
                done += batch.num_rows
                log.info("  %s: %d / %d (%.0fs)", table, done, expected, time.monotonic() - t0)
        conn.commit()

        for ddl in spec["indexes"]:
            log.info("  %s", ddl)
            cur.execute(ddl)
        conn.commit()

        if spec.get("cluster"):
            # рядки одного треку лежать поруч -> профіль/дати читаються кількома сторінками, а не по всьому диску
            log.info("  CLUSTER %s USING %s", table, spec["cluster"])
            cur.execute(f"CLUSTER {table} USING {spec['cluster']}")
            conn.commit()

        cur.execute(f"ANALYZE {table}")
        conn.commit()

        cur.execute(f"SELECT count(*) FROM {table}")
        got = cur.fetchone()[0]

    if got != expected:
        raise RuntimeError(f"{table}: rows mismatch, parquet={expected} postgis={got}")
    log.info("%s OK: %d rows in %.0fs", table, got, time.monotonic() - t0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=sorted(TABLES), help="завантажити лише одну таблицю")
    args = ap.parse_args()

    keys = [args.only] if args.only else ["nmad", "points"]
    with psycopg.connect(_conninfo()) as conn:
        for k in keys:
            load(conn, TABLES[k])
    return 0


if __name__ == "__main__":
    sys.exit(main())
