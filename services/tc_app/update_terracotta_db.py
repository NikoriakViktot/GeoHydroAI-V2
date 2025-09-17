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

# --- Додати дані в Terracotta + зібрати index
with driver.connect():
    for record in find_files(GEOTIFF_DIR):
        try:
            driver.insert({"category": record["category"], "name": record["name"]}, record["path"])
            print(f"✅ inserted: {record['category']}/{record['name']}")
            index_list.append(record)
        except Exception as e:
            print(f"❌ failed: {record['name']} — {e}")

# --- Зберегти metadata.json для Dash
with open("metadata.json", "w") as f:
    json.dump(index_list, f, indent=2)
    print("📦 metadata.json оновлено")
