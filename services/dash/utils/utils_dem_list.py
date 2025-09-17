#utils/utils_dem_list.py

import json

def get_dem_options(index_path="assets/metadata.json"):
    with open(index_path, "r") as f:
        index = json.load(f)
    # Фільтруємо лише DEM основні
    dems = [rec for rec in index if rec["category"] == "dem"]
    return [{"label": dem["name"], "value": dem["path"]} for dem in dems]
