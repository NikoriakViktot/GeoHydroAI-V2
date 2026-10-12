#services/terracotta/update_terracotta_db.py

import os
import json
from terracotta import get_driver
from config import GEOTIFF_DIR, DRIVER_PATH, PROJECTION

driver = get_driver(str(DRIVER_PATH))
index_list = []

def parse_layer_name(name):
    parts = name.split("_")
    dem = "_".join(parts[:2])  # alos_dem

    hand = None
    flood = None

    if "hand" in parts:
        hand_index = parts.index("hand")
        try:
            hand = f"{parts[hand_index]}_{parts[hand_index + 1]}"
        except IndexError:
            hand = parts[hand_index]  # просто "hand"

    if "flood" in parts:
        flood_index = parts.index("flood")
        try:
            flood = parts[flood_index + 1]  # "10m", "1m", ...
        except IndexError:
            flood = None

    return dem, hand, flood

def clean_name(fname: str):
    return fname.replace("_cog.tif", "").replace(".tif", "").replace("_utm32635", "")

def extract_tags(name):
    return name.lower().split("_")

def find_files(base_dir):
    for root, _, files in os.walk(base_dir):
        for fname in files:
            if not fname.endswith(".tif"):
                continue
            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, base_dir)
            parts = rel_path.split(os.sep)

            file = parts[-1]
            category = os.path.join(*parts[:-1]) if len(parts) > 1 else "uncategorized"
            name = clean_name(file)
            dem, hand, flood = parse_layer_name(name)

            yield {
                "category": category,
                "name": name,
                "path": full_path,
                "tags": extract_tags(name),
                "projection": PROJECTION,
                "dem": dem,
                "hand": hand,
                "flood": flood
            }


# --- Створити БД, якщо немає
if not DRIVER_PATH.exists():
    os.makedirs(DRIVER_PATH.parent, exist_ok=True)
    driver.create(keys=["category", "name"])

# Terracotta 0.8 computes a dataset's statistics by reading the whole raster into memory (crick is not installed).
# The national 30 m COGs are ~4.5 GB decompressed and killed the container (768 MB) in a restart loop on 2026-10-12.
# Large files therefore get their metadata from an overview (max_shape), which is what tile colouring needs.
# Files whose size and mtime did not change since the last start are not re-inserted (fast restarts); the state is
# kept next to the sqlite database.
# by pixel count, not file size: a compressed Byte COG of 49 MB can hold 4.5 billion pixels
LARGE_PIXELS = int(os.getenv("TC_LARGE_PIXELS", str(50_000_000)))
MAX_SHAPE = (4096, 4096)
STATE = DRIVER_PATH.parent / "indexed.json"
try:
    state = json.loads(STATE.read_text())
except (OSError, ValueError):
    state = {}
reindex = os.getenv("TC_REINDEX") == "1"
known = set(driver.get_datasets().keys()) if not reindex else set()
new_state = {}

# --- Додати дані в Terracotta + зібрати index
with driver.connect():
    for record in find_files(GEOTIFF_DIR):
        try:
            st = os.stat(record["path"])
            sig = f"{st.st_size}:{st.st_mtime_ns}"
            key = (record["category"], record["name"])
            new_state[record["path"]] = sig
            index_list.append(record)
            if not reindex and state.get(record["path"]) == sig and key in known:
                continue
            metadata = None
            import rasterio
            with rasterio.open(record["path"]) as src:
                large = src.width * src.height > LARGE_PIXELS
            if large:
                from terracotta import raster
                metadata = raster.compute_metadata(record["path"], max_shape=MAX_SHAPE)
            driver.insert({"category": record["category"], "name": record["name"]}, record["path"], metadata=metadata)
            print(f"✅ inserted: {record['category']}/{record['name']}" + (" (metadata from an overview)" if metadata else ""), flush=True)
        except Exception as e:
            print(f"❌ failed: {record['name']} — {e}", flush=True)

try:
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(new_state))
    os.replace(tmp, STATE)
except OSError as e:
    print(f"⚠️ index state not saved: {e}")

# --- Зберегти metadata.json для Dash
with open("metadata.json", "w") as f:
    json.dump(index_list, f, indent=2)
    print("📦 metadata.json оновлено")
