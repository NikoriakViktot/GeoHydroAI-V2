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
    Повертає (diff[np.float32], DEM1_as_reference)
    """
    dem1 = DEM(path1)  # має data, transform, profile (crs, nodata, ...)
    nodata1 = dem1.profile.get("nodata", None)

    with rasterio.open(path2) as src2:
        dem2_data = src2.read(1, masked=False)
        nodata2 = src2.nodata

        # 1) Ресемплінг DEM2 у грід DEM1 з явним опрацюванням nodata
        resampled = np.full_like(dem1.data, fill_value=np.nan, dtype=np.float32)
        reproject(
            source=dem2_data,
            destination=resampled,
            src_transform=src2.transform,
            src_crs=src2.crs,
            dst_transform=dem1.transform,
            dst_crs=dem1.profile["crs"],
            src_nodata=nodata2,
            dst_nodata=np.nan,               # важливо: пишемо NaN у місця без даних
            resampling=Resampling.bilinear,
        )

    # 2) Різниця
    diff = resampled.astype(np.float32) - dem1.data.astype(np.float32)

    # 3) Комбінована маска: обидва nodata + нечислові + «абсурдні» висоти/різниці
    mask = np.zeros(diff.shape, dtype=bool)

    # nodata/NaN з DEM1
    if nodata1 is not None:
        mask |= (dem1.data == nodata1) | np.isclose(dem1.data, nodata1)
    mask |= ~np.isfinite(dem1.data)

    # nodata/NaN з DEM2 (після ресемплу ми вже маємо NaN)
    if nodata2 is not None:
        mask |= (resampled == nodata2) | np.isclose(resampled, nodata2)
    mask |= ~np.isfinite(resampled)

    # «Абсурдні» значення висот/різниці (страхувальна огорожа)
    # (типові деми в межах ±9000 м; також відкинемо екстремальний diff)
    mask |= (dem1.data < -9000) | (dem1.data > 9000)
    mask |= (resampled < -9000) | (resampled > 9000)
    mask |= (diff < -9000) | (diff > 9000)

    diff = np.where(mask, np.nan, diff).astype(np.float32)
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

from rasterio.warp import transform as rio_transform

def raster_bounds_ll(ref_dem):
    """
    [[south, west], [north, east]] у WGS84 — для BitmapLayer/Leaflet.
    """
    with rasterio.open(ref_dem.filename) as src:
        b = src.bounds
        crs = src.crs
        # 4 кути в CRS растра -> WGS84
        xs = [b.left,  b.right, b.right, b.left]
        ys = [b.bottom, b.bottom, b.top,   b.top]
        lon, lat = rio_transform(crs, "EPSG:4326", xs, ys)
        # обернемо в bbox
        west, east = min(lon), max(lon)
        south, north = min(lat), max(lat)
        return [[south, west], [north, east]]