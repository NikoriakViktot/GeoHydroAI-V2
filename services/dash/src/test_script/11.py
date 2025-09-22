import os, json, logging

TC_BASE = os.getenv("TERRACOTTA_PUBLIC_URL", "https://www.geohydroai.org/tc").rstrip("/")
ASSETS_INDEX_PATH = r"C:\Users\5302\PycharmProjects\GeoHydroAI-V2\services\dash\assets\layers_index.json"

def _fix_path(p: str) -> str:
    if not p: return p
    p = p.replace("\\", "/")
    if p.startswith("/"): return os.path.normpath(p)
    if p.startswith("data/COG/"):  return "/app/data/cogs/" + p.split("data/COG/")[1]
    if p.startswith("data/cogs/"): return "/app/data/cogs/" + p.split("data/cogs/")[1]
    if p.startswith("data/"):      return "/app/" + p
    return os.path.normpath(p)

def _parse_level(s: str) -> int:
    try: return int(str(s).lower().replace("m", "").strip())
    except Exception: return 0

def build_dem_url(dem_name: str, cmap: str, stretch) -> str:
    s = f"[{stretch[0]},{stretch[1]}]"
    return f"{TC_BASE}/singleband/dem/{dem_name}" + "/{z}/{x}/{y}.png" + f"?colormap={cmap}&stretch_range={s}"

def build_flood_url(dem_name: str, hand_name: str, level: str, cmap: str, stretch, pure_blue: bool) -> str:
    layer = f"{dem_name}_{hand_name}_flood_{level}"
    s = f"[{stretch[0]},{stretch[1]}]"
    base = f"{TC_BASE}/singleband/flood_scenarios/{layer}" + "/{z}/{x}/{y}.png"
    return f"{base}?colormap=custom&colors=0000ff&stretch_range={s}" if pure_blue \
           else f"{base}?colormap={cmap}&stretch_range={s}"

# ---------- 1) читаємо та нормалізуємо індекс ----------
with open(ASSETS_INDEX_PATH, "r") as f:
    raw_index = json.load(f)
items = raw_index if isinstance(raw_index, list) else [raw_index]
layers_index = []
for rec in items:
    r = dict(rec)
    if r.get("path"):
        r["path"] = _fix_path(r["path"])
    layers_index.append(r)

print(f"TC_BASE = {TC_BASE}")
print(f"layers_index entries = {len(layers_index)}")

# ---------- 2) збираємо DEM → levels та (DEM, level) → hand ----------
dem_levels = {}                 # dem -> sorted list of levels
dem_level2hand = {}             # dem -> { level -> chosen hand }

tmp_levels = {}
tmp_level2hand = {}
for r in layers_index:
    if r.get("category") != "flood_scenarios":
        continue
    dem, hand, level = r.get("dem"), r.get("hand"), r.get("flood")
    if not (dem and hand and level):
        continue
    tmp_levels.setdefault(dem, set()).add(level)
    tmp_level2hand.setdefault((dem, level), set()).add(hand)

for dem, lvlset in tmp_levels.items():
    lvls = sorted(lvlset, key=_parse_level)
    dem_levels[dem] = lvls
    dem_level2hand[dem] = {}
    for L in lvls:
        hands = list(tmp_level2hand.get((dem, L), []))
        chosen = "hand_2000" if "hand_2000" in hands else (hands[0] if hands else "")
        dem_level2hand[dem][L] = chosen

print("DEMs:", ", ".join(sorted(dem_levels.keys())))
for dem in sorted(dem_levels.keys()):
    print(f"  {dem}: levels={dem_levels[dem]}  | hand per level={ {L: dem_level2hand[dem][L] for L in dem_levels[dem]} }")

# ---------- 3) будуємо прикладові URL-и ----------
# Вибираємо будь-який доступний DEM та рівень
test_dem = next(iter(sorted(dem_levels.keys())), None)
if test_dem:
    levels = dem_levels[test_dem]
    test_level = "5m" if "5m" in levels else levels[0]
    test_hand = dem_level2hand[test_dem][test_level]

    dem_url   = build_dem_url(test_dem, "terrain", [250, 2200])
    flood_url = build_flood_url(test_dem, test_hand, test_level, "blues", [0, 5], pure_blue=False)

    print("\nSample URLs:")
    print("  DEM  ->", dem_url)
    print("  FLOOD->", flood_url)
else:
    print("No DEMs found in index.")

# ---------- 4) (опційно) перевірити HTTP 1 тайл Terracotta ----------
# Працює тільки якщо контейнер може дістатися до TC_BASE і шар реально опублікований там.
try:
    import requests
    url_test = flood_url.replace("{z}", "10").replace("{x}", "600").replace("{y}", "400")
    r = requests.get(url_test, timeout=5)
    print(f"\nHTTP test GET {url_test}\n  status={r.status_code} content-type={r.headers.get('Content-Type')}")
except Exception as e:
    print(f"(skip http test) {e}")
