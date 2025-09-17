#utils/dem_tools.py
import os
import uuid
import numpy as np
import matplotlib.pyplot as plt
import base64
import io
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.stats import skew, kurtosis
from xdem import DEM

def compute_dem_difference(path1, path2):
    """
    Повертає різницю DEM (DEM2 - DEM1), автоматично ресемплить/реєструє під grid DEM1.
    """
    dem1 = DEM(path1)
    with rasterio.open(path2) as src:
        dem2_data = src.read(1)
        # Якщо CRS не збігається, кинемо помилку (або додати автоматичний reprojection)
        if src.crs != dem1.profile['crs']:
            raise ValueError("CRS mismatch: reprojection for different CRS не реалізовано в цьому коді!")
        # Ресемплінг DEM2 під DEM1
        resampled_dem2 = np.empty_like(dem1.data)
        reproject(
            source=dem2_data,
            destination=resampled_dem2,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dem1.transform,
            dst_crs=dem1.profile['crs'],
            resampling=Resampling.bilinear
        )
    nodata = dem1.profile.get('nodata', -9999)
    diff = resampled_dem2 - dem1.data
    diff = np.where((diff == nodata) | (diff < -9000), np.nan, diff)
    return diff, dem1

def save_temp_diff_as_cog(diff_array, reference_dem, prefix="diff_"):
    filename = f"{prefix}{uuid.uuid4().hex}.tif"
    out_path = os.path.join("/tmp", filename)
    profile = reference_dem.profile
    profile.update({
        "driver": "COG",
        "dtype": "float32",
        "count": 1,
        "compress": "deflate"
    })
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(diff_array.astype("float32"), 1)
    return out_path

def save_diff_png(diff_array, ref_dem, out_path, vmin=-10, vmax=10):
    # Збереження PNG з extent
    with rasterio.open(ref_dem.filename) as src:
        bounds = src.bounds
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    plt.figure(figsize=(7, 9))
    plt.imshow(diff_array, cmap='RdBu', vmin=vmin, vmax=vmax, extent=extent, origin='upper')
    plt.axis('off')
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

def save_colorbar_png(vmin, vmax, cmap, out_path, label="ΔH (m)"):
    import matplotlib as mpl
    fig, ax = plt.subplots(figsize=(2, 5))
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='vertical')
    cb.set_label(label, fontsize=13)
    plt.savefig(out_path, bbox_inches='tight', transparent=True, pad_inches=0.1)
    plt.close()

def plot_histogram(diff_array, bins=50, clip_range=(-50, 50)):
    valid = diff_array[~np.isnan(diff_array)]
    # Фільтруємо "реалістичні" значення
    valid = valid[(valid > clip_range[0]) & (valid < clip_range[1])]
    if len(valid) == 0:
        # Якщо після фільтрації немає даних — повертаємо пусту картинку
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title("No data after clipping")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight')
        plt.close(fig)
        img_base64 = base64.b64encode(buf.getvalue()).decode('ascii')
        return f"data:image/png;base64,{img_base64}"
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(valid, bins=bins, color="skyblue", edgecolor="black")
    ax.set_title("Histogram of Elevation Errors")
    ax.set_xlabel("Error (m)")
    ax.set_ylabel("Frequency")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig)
    img_base64 = base64.b64encode(buf.getvalue()).decode('ascii')
    return f"data:image/png;base64,{img_base64}"



def get_raster_bounds(raster_path):
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        # Leaflet expects [[south, west], [north, east]]
        return [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]

def calculate_error_statistics(diff_array: np.ndarray):
    valid = diff_array[~np.isnan(diff_array)]
    if valid.size == 0:
        return {
            "count": 0,
            "mean_error": np.nan,
            "median_error": np.nan,
            "std_dev": np.nan,
            "rmse": np.nan,
            "min": np.nan,
            "max": np.nan,
            "skewness": np.nan,
            "kurtosis": np.nan,
        }
    return {
        "count": int(valid.size),
        "mean_error": float(np.mean(valid)),
        "median_error": float(np.median(valid)),
        "std_dev": float(np.std(valid)),
        "rmse": float(np.sqrt(np.mean(valid ** 2))),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "skewness": float(skew(valid)),
        "kurtosis": float(kurtosis(valid))
    }
