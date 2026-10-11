"""Print a version per Terracotta category, derived from its files, as JSON.

The frontend appends ?v=<version> to raster tile URLs; nginx caches a versioned URL as immutable. When any
file of a category is replaced (new size or mtime) or a dataset is added or removed, the category gets a
new version and browsers and the nginx tile cache fetch fresh tiles; everything else stays cached.

Runs inside the terracotta container (it reads TC_DB and sees the COG paths stored in it):
    docker exec -i geoai-terracotta-1 python - < services/tc_app/scripts/raster_versions.py
"""
import hashlib
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone

db = os.environ.get("TC_DB", "/app/tc_db/terracotta.sqlite")
rows = sqlite3.connect(f"file:{db}?mode=ro", uri=True).execute("SELECT category, name, path FROM datasets").fetchall()

parts: dict[str, list[str]] = {}
missing = 0
for category, name, path in sorted(rows):
    try:
        st = os.stat(path)
        parts.setdefault(category, []).append(f"{name}:{st.st_size}:{st.st_mtime_ns}")
    except FileNotFoundError:
        missing += 1
        parts.setdefault(category, []).append(f"{name}:missing")

versions = {c: hashlib.sha1("|".join(p).encode()).hexdigest()[:10] for c, p in parts.items()}
json.dump({"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "datasets": len(rows),
           "missing_files": missing, "versions": versions}, sys.stdout, indent=1, sort_keys=True)
