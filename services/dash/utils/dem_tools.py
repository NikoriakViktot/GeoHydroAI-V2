# utils/dem_tools.py
import os, uuid, io, base64
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.stats import skew, kurtosis
from xdem import DEM


def compute_dem_difference(path1, path2):
    """
    diff = DEM2 - DEM1, DEM2 автоматично ресемплиться/репроєктується під grid DEM1.
    Повертає (diff_array[np.float32], reference_dem(DEM))
    """
    dem1 = DEM(path1)  # має data, transform, profile (crs, nodata, ...)
    with rasterio.open(path2) as src2:
        dem2 = src2.read(1, masked=False)
        resampled = np.empty_like(dem1.data, dtype=np.float32)
        reproject(
            source=dem2,
            destination=resampled,
            src_transform=src2.transform,
            src_crs=src2.crs,
            dst_transform=dem1.transform,
            dst_crs=dem1.profile["crs"],
            resampling=Resampling.bilinear,
        )

    nodata = dem1.profile.get("nodata", -9999)
    diff = resampled.astype(np.float32) - dem1.data.astype(np.float32)
    diff = np.where(
        (dem1.data == nodata) | np.isclose(dem1.data, nodata) | (diff < -9000) | (diff > 9000),
        np.nan,
        diff,
    ).astype(np.float32)
    return diff, dem1


def save_temp_diff_as_cog(diff_array, reference_dem, prefix="demdiff_"):
    """
    Пише COG у /tmp (driver=COG). Використовується лише якщо потрібні тайли з сервера.
    """
    import rasterio
    filename = f"{prefix}{uuid.uuid4().hex}.tif"
    out_path = os.path.join("/tmp", filename)
    profile = reference_dem.profile.copy()
    profile.update({"driver": "COG", "dtype": "float32", "count": 1, "compress": "deflate"})
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(diff_array.astype("float32"), 1)
    return out_path


def make_colorbar_datauri(vmin, vmax, cmap="RdBu_r", label="ΔH (m)"):
    import matplotlib as mpl
    fig, ax = plt.subplots(figsize=(2, 5))
    # темний фон під легенду
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation="vertical")
    cb.set_label(label, fontsize=12, color="#eee")
    cb.ax.tick_params(colors="#eee")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", transparent=True, pad_inches=0.1)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def plot_histogram(diff_array, bins=60, clip_range=(-50, 50)):
    """
    Гістограма в ЧОРНІЙ темі (dark): темний фон, світлі підписи.
    Повертає data:image/png;base64,...
    """
    valid = diff_array[~np.isnan(diff_array)]
    valid = valid[(valid > clip_range[0]) & (valid < clip_range[1])]

    fig, ax = plt.subplots(figsize=(8, 3.6))
    # чорна тема
    fig.patch.set_facecolor("#0e0e0e")
    ax.set_facecolor("#0e0e0e")
    [sp.set_color("#bbb") for sp in ax.spines.values()]
    ax.tick_params(colors="#ddd")
    ax.title.set_color("#ddd")
    ax.xaxis.label.set_color("#ddd")
    ax.yaxis.label.set_color("#ddd")

    if valid.size == 0:
        ax.set_title("No data after clipping")
        ax.axis("off")
    else:
        ax.hist(valid, bins=bins, color="#6aa9ff", edgecolor="#e6f0ff")
        ax.set_title("Histogram of Elevation Errors")
        ax.set_xlabel("Error (m)")
        ax.set_ylabel("Frequency")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def get_raster_bounds(raster_path):
    with rasterio.open(raster_path) as src:
        b = src.bounds
        return [[b.bottom, b.left], [b.top, b.right]]


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
        "kurtosis": float(kurtosis(valid)),
    }


# ---------- overlay як data-URI + bounds для deck.gl ----------
# utils/dem_tools.py
import io, base64, matplotlib.pyplot as plt
import rasterio

def diff_to_base64_png(diff_array, ref_dem, vmin=-10, vmax=10, figsize=(8, 8)):
    """Рендер PNG (прозорий) для накладання як BitmapLayer. Повертає data:image/png;base64,..."""
    with rasterio.open(ref_dem.filename) as src:
        b = src.bounds
    extent = [b.left, b.right, b.bottom, b.top]
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        diff_array, cmap="RdBu_r", vmin=vmin, vmax=vmax,
        extent=extent, origin="upper"
    )
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

def raster_bounds_ll(ref_dem):
    """[[south, west], [north, east]] для географічних bounds BitmapLayer/Leaflet."""
    with rasterio.open(ref_dem.filename) as src:
        b = src.bounds
        return [[b.bottom, b.left], [b.top, b.right]]
