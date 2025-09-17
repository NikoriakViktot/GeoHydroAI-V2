import os
import subprocess
from pathlib import Path
from shutil import copyfile

def detect_resampling_strategy(filename: str) -> str:
    fname = filename.lower()
    if any(key in fname for key in ["hand", "stream", "lulc", "class", "zone"]):
        return "near"
    elif "geomorph" in fname:
        return "mode"
    elif any(key in fname for key in ["aspect", "slope_horn", "tpi", "tri", "curvature", "roughness"]):
        return "bilinear"
    else:
        return "bilinear"

def convert_all_to_cog(input_dir: str, output_dir: str):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    all_tifs = list(input_dir.rglob("*.tif"))
    print(f"[📦] Знайдено {len(all_tifs)} TIFF-файлів у {input_dir}")

    for file in all_tifs:
        rel_path = file.relative_to(input_dir)
        out_subdir = output_dir / rel_path.parent
        out_subdir.mkdir(parents=True, exist_ok=True)

        resampling = detect_resampling_strategy(file.name)
        print(f"\n📂 Обробка: {rel_path} (метод: {resampling})")

        temp_warp = out_subdir / f"{file.stem}_3857_temp.tif"
        cog_out = out_subdir / f"{file.stem}_cog.tif"

        # 1. gdalwarp до EPSG:3857
        subprocess.run([
            "gdalwarp",
            "-t_srs", "EPSG:3857",
            "-r", resampling,
            "-co", "TILED=YES",
            str(file),
            str(temp_warp)
        ], check=True)

        # 2. gdal_translate → COG
        subprocess.run([
            "gdal_translate",
            "-of", "COG",
            "-co", "COMPRESS=DEFLATE",
            str(temp_warp),
            str(cog_out)
        ], check=True)

        temp_warp.unlink()
        print(f"✅ Збережено: {cog_out.relative_to(output_dir)}")

    print("\n🏁 Усі файли перетворено в COG")

# === Приклад використання ===
if __name__ == "__main__":
    convert_all_to_cog(
        input_dir="/mnt/c/Users/5302/OneDrive/PhD/paper_DEM_artickle/data/DATA_UTM/",
        output_dir="/mnt/c/Users/5302/OneDrive/PhD/paper_DEM_artickle/data/COG/"
    )
