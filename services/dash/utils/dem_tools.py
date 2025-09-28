# utils/dem_tools.py
import os, uuid, io, base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl

import rasterio
from rasterio.warp import transform as rio_transform
from rasterio.warp import reproject, Resampling
from scipy.stats import skew, kurtosis
from xdem import DEM
import pandas as pd
import plotly.graph_objs as go
import xdem


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
    filename = f"{prefix}{uuid.uuid4().hex}.tif"
    out_path = os.path.join("/tmp", filename)
    profile = reference_dem.profile.copy()
    profile.update({"driver": "COG", "dtype": "float32", "count": 1, "compress": "deflate"})
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(diff_array.astype("float32"), 1)
    return out_path

def make_colorbar_datauri(
    vmin, vmax,
    cmap="RdBu_r",                 # можна спробувати: "turbo", "seismic"
    label="ΔH (m)",
    sat_boost=1.25,                # множник насиченості (S у HSV)
    val_boost=1.05,                # трохи піднімемо яскравість (V)
    gamma=0.9,                     # <1 підсилює краї (PowerNorm)
    center=None,                   # для різниць постав center=0 і отримаєш двосхилу норм.
    dpi=300
):

    # 1) беремо базову палітру і підвищуємо насиченість/яскравість
    base = mpl.cm.get_cmap(cmap, 256)
    colors = base(np.linspace(0, 1, 256))
    rgb = colors[:, :3]
    hsv = mpl.colors.rgb_to_hsv(rgb[np.newaxis, ...])[0]
    hsv[:, 1] = np.clip(hsv[:, 1] * sat_boost, 0, 1)
    hsv[:, 2] = np.clip(hsv[:, 2] * val_boost, 0, 1)
    colors[:, :3] = mpl.colors.hsv_to_rgb(hsv[np.newaxis, ...])[0]
    cmap_boost = mpl.colors.ListedColormap(colors)

    # 2) нормалізація: або степенева (контраст на краях), або двосхила з центром
    if center is None:
        norm = mpl.colors.PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)
    else:
        norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)

    # 3) рендер
    fig, ax = plt.subplots(figsize=(2, 5), dpi=dpi)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap_boost, norm=norm, orientation="vertical")
    cb.set_label(label, fontsize=12, color="#eee")
    cb.ax.tick_params(colors="#eee")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", transparent=True, pad_inches=0.1)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")



# def make_colorbar_datauri(vmin, vmax, cmap="RdBu_r", label="ΔH (m)"):
#     fig, ax = plt.subplots(figsize=(2, 5))
#     # темний фон під легенду
#     fig.patch.set_alpha(0.0)
#     ax.set_facecolor("none")
#     norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
#     cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation="vertical")
#     cb.set_label(label, fontsize=12, color="#eee")
#     cb.ax.tick_params(colors="#eee")
#     buf = io.BytesIO()
#     fig.savefig(buf, format="png", bbox_inches="tight", transparent=True, pad_inches=0.1)
#     plt.close(fig)
#     return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def plot_histogram(diff_array, bins=60, clip_range=(-50, 50)):
    """
    Гістограма в ЧОРНІЙ темі (dark): темний фон, світлі підписи.
    Повертає data:image/png;base64,...
    """
    valid = diff_array[~np.isnan(diff_array)]
    valid = valid[(valid > clip_range[0]) & (valid < clip_range[1])]

    fig, ax = plt.subplots(figsize=(8, 4.5))
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


def _nmad(a, axis=None, keepdims=False, return_median=False):
    """
    Normalized Median Absolute Deviation (NMAD), NaN-safe.

    Parameters
    ----------
    a : array_like
    axis : int | tuple[int] | None
    keepdims : bool
    return_median : bool
        Якщо True — повертає (nmad, median).

    Returns
    -------
    nmad : float | ndarray
    median : float | ndarray  (опційно)
    """
    a = np.asanyarray(a)
    # медіана з пропуском NaN
    med = np.nanmedian(a, axis=axis, keepdims=True)
    # |x - median|
    mad = np.nanmedian(np.abs(a - med), axis=axis, keepdims=keepdims)
    nmad_val = 1.4826 * mad  # шкалування під нормальний розподіл

    # якщо всі NaN по вибранiй осі — np.nanmedian поверне NaN; це те, що зазвичай треба
    if return_median:
        return nmad_val, (med if keepdims else np.squeeze(med, axis=axis))
    return nmad_val

def robust_stats(diff: np.ndarray, clip=None):
    v = diff[np.isfinite(diff)]
    if v.size == 0:
        return {
            "count": 0, "median": np.nan, "nmad": np.nan, "mae": np.nan, "rmse": np.nan,
            "mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan,
            "p5": np.nan, "p95": np.nan, "outlier_%": np.nan, "skew": np.nan, "kurtosis": np.nan
        }
    if clip is not None:
        lo, hi = np.nanpercentile(v, clip)
        v = v[(v >= lo) & (v <= hi)]
    p5, p95 = np.nanpercentile(v, [5, 95])
    T = 3 * _nmad(v)  # поріг аутлаєрів
    outlier = np.sum(np.abs(v - np.nanmedian(v)) > T) / v.size * 100.0
    return {
        "count": int(v.size),
        "median": float(np.nanmedian(v)),
        "nmad": float(_nmad(v)),
        "mae": float(np.nanmean(np.abs(v))),
        "rmse": float(np.sqrt(np.nanmean(v**2))),
        "mean": float(np.nanmean(v)),
        "std": float(np.nanstd(v)),
        "min": float(np.nanmin(v)),
        "max": float(np.nanmax(v)),
        "p5": float(p5),
        "p95": float(p95),
        "outlier_%": float(outlier),
        "skew": float(skew(v, nan_policy="omit")),
        "kurtosis": float(kurtosis(v, nan_policy="omit")),
    }


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

def xdem_uncertainty(diff_arr, ref_dem, stable_mask=None, random_state=42):
    """
    Повертає:
      - err_map: карта 1σ випадкової помилки (м)
      - params: параметри моделі варіограми
      - rho: функція кореляції rho(h) (callable)
      - stderr_mean: 1σ похибка середнього по всій області diff_arr (спрощено)
    """
    try:
        # атрибути рельєфу
        slope, maxc = xdem.terrain.get_terrain_attribute(ref_dem, attribute=["slope","maximum_curvature"])
        # гетероскедастичність
        errors, df_bin, err_fn = xdem.spatialstats.infer_heteroscedasticity_from_stable(
            dvalues=diff_arr, list_var=[slope, maxc], list_var_names=["slope","maxc"],
            unstable_mask=None if stable_mask is None else ~stable_mask
        )
        # стандартизація та варіограма
        z = diff_arr / errors
        emp_vgm, params, rho = xdem.spatialstats.infer_spatial_correlation_from_stable(
            dvalues=z, list_models=["Gaussian","Spherical"], random_state=random_state
        )
        # груба оцінка похибки середнього по всій області
        mean_sig = float(np.nanmean(errors))
        # ефективна к-ть зразків (спрощено за площею растру без векторів)
        n_eff = np.isfinite(diff_arr).sum()
        stderr_mean = mean_sig / np.sqrt(max(1, n_eff))
        return {"err_map": getattr(errors, "data", errors),
                "params": params, "rho": rho, "stderr_mean": float(stderr_mean)}
    except Exception as e:
        return {"error": str(e)}

# ---------- overlay як data-URI + bounds для deck.gl ----------


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
# ---- flood alignment helpers ----


def read_binary_with_meta(path: str):
    """Читає бінарний GeoTIFF і повертає (mask_bool, transform, crs, width, height, nodata)."""
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        nodata = ds.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, 0, arr)
        return (arr.astype(bool), ds.transform, ds.crs, ds.width, ds.height, nodata)

def _transform_equal(t1, t2, rtol=1e-6, atol=1e-9):
    """Порівняння Affine з допуском (бо double не завжди біт-в-біт)."""
    a = np.array([t1.a, t1.b, t1.c, t1.d, t1.e, t1.f])
    b = np.array([t2.a, t2.b, t2.c, t2.d, t2.e, t2.f])
    return np.allclose(a, b, rtol=rtol, atol=atol)

def reproject_to_grid(src_bool: np.ndarray,
                      src_transform, src_crs,
                      dst_width: int, dst_height: int,
                      dst_transform, dst_crs):
    """Ресемплінг у цільову сітку (nearest для бінарних шарів)."""
    dst_u8 = np.zeros((dst_height, dst_width), dtype=np.uint8)
    reproject(
        source=src_bool.astype(np.uint8),
        destination=dst_u8,
        src_transform=src_transform, src_crs=src_crs,
        dst_transform=dst_transform, dst_crs=dst_crs,
        resampling=Resampling.nearest,
        src_nodata=0, dst_nodata=0
    )
    return dst_u8.astype(bool)

def align_boolean_pair(A_bool, A_tx, A_crs, A_w, A_h,
                       B_bool, B_tx, B_crs, B_w, B_h):
    """Повертає (A_aligned, B_aligned) на одній сітці (сітка A)."""
    same_grid = (A_w == B_w and A_h == B_h and
                 (A_crs == B_crs) and _transform_equal(A_tx, B_tx))
    if same_grid:
        return A_bool, B_bool

    B_aligned = reproject_to_grid(
        B_bool, B_tx, B_crs,
        A_w, A_h, A_tx, A_crs
    )
    return A_bool, B_aligned

def crop_to_common_extent(A_bool, B_bool):
    """Підстраховка: якщо розміри іще різняться на 1–2 пікс — обрізаємо до мінімуму."""
    h = min(A_bool.shape[0], B_bool.shape[0])
    w = min(A_bool.shape[1], B_bool.shape[1])
    return A_bool[:h, :w], B_bool[:h, :w]

# --- Plotly: гістограма для dH (щоб не PNG) ---
def plotly_histogram_figure(diff_array, bins=60, clip_range=(-50, 50), density=False, cumulative=False, title="Histogram of Elevation Errors"):
    vals = diff_array[np.isfinite(diff_array)]
    vals = vals[(vals > clip_range[0]) & (vals < clip_range[1])]
    fig = go.Figure()
    if vals.size == 0:
        fig.update_layout(template="plotly_dark",
                          annotations=[dict(text="No data after clipping", showarrow=False, x=0.5, y=0.5)])
        fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
        return fig
    counts, edges = np.histogram(vals, bins=bins, density=density)
    centers = 0.5*(edges[:-1]+edges[1:])
    y = np.cumsum(counts) if cumulative else counts
    if cumulative and density and y[-1] > 0: y = y / y[-1]
    fig.add_bar(x=centers, y=y, name=("Cumulative" if cumulative else "Count"))
    fig.update_layout(template="plotly_dark", title=title, xaxis_title="Error (m)",
                      yaxis_title=("Cumulative density" if (cumulative and density) else ("Cumulative count" if cumulative else ("Density" if density else "Frequency"))),
                      margin=dict(l=8,r=8,t=28,b=38), bargap=0.05)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    fig.update_layout(margin=dict(l=40, r=10, t=24, b=40), bargap=0.05)
    return fig


# --- Plotly: ECDF для dH (НОВА ФУНКЦІЯ) ---
def plotly_ecdf_figure(diff_array, clip_range=(-50, 50), title="ECDF of Elevation Errors (dH)"):
    """
    Побудова графіка Емпіричної кумулятивної функції розподілу (ECDF)
    для різниці висот (dH) з позначенням 5-го та 95-го перцентилів.
    """
    vals = diff_array[np.isfinite(diff_array)]
    vals = vals[(vals > clip_range[0]) & (vals < clip_range[1])]

    fig = go.Figure()
    if vals.size == 0:
        # Обробка випадку, коли немає даних (копіюємо стиль з histogram)
        fig.update_layout(template="plotly_dark",
                          annotations=[dict(text="No data after clipping", showarrow=False, x=0.5, y=0.5)])
        fig.update_xaxes(visible=False);
        fig.update_yaxes(visible=False)
        return fig

    # 1. Сортуємо дані
    sorted_vals = np.sort(vals)

    # 2. Розраховуємо кумулятивну ймовірність: ECDF = (i + 1) / N
    ecdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

    # 3. Додаємо лінію ECDF
    fig.add_trace(go.Scatter(
        x=sorted_vals,
        y=ecdf,
        mode='lines',
        name='ECDF',
        line=dict(color="#1f77b4", width=2),
        hovertemplate='Error (m): %{x:.2f}<br>Probability (P(X ≤ x)): %{y:.3f}<extra></extra>'
    ))

    # 4. Додаємо маркери для 5-го та 95-го перцентилів
    p5, p95 = np.nanpercentile(vals, [5, 95])

    fig.add_trace(go.Scatter(
        x=[p5, p95],
        y=[0.05, 0.95],
        mode='markers',
        marker=dict(size=8, color='gold', symbol='circle'),
        name=f'P5/P95 ({p5:.2f}/{p95:.2f} m)',
        hovertemplate='Percentile: %{y}<br>Error (m): %{x:.2f}<extra></extra>'
    ))

    # 5. Оновлюємо вигляд (темна тема)
    fig.update_layout(
        template="plotly_dark",
        title=title,
        xaxis_title="Error (m)",
        yaxis_title="Cumulative Probability (P(X ≤ x))",
        margin=dict(l=40, r=10, t=24, b=40),
        hovermode="x unified",
        legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99)
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)", range=[0, 1.0])  # Range від 0 до 1

    return fig
# --- Plotly: violin plot для dH ---
def plotly_violin_figure(diff_array, clip_range=None, title="Distribution (Violin)"):
    """
    Одновимірний violin plot для dH з опційним кліпом по (vmin, vmax).
    Повертає plotly Figure у темній темі, з відображенням box та meanline.
    """
    vals = diff_array[np.isfinite(diff_array)].ravel()
    if clip_range is not None:
        vmin, vmax = clip_range
        vals = vals[(vals >= vmin) & (vals <= vmax)]

    fig = go.Figure()
    if vals.size == 0:
        fig.update_layout(
            template="plotly_dark",
            annotations=[dict(text="No data after clipping", showarrow=False, x=0.5, y=0.5)]
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        return fig

    fig.add_trace(go.Violin(
        y=vals,
        box_visible=True,
        meanline_visible=True,
        points="outliers",   # показувати лише викиди
        opacity=0.9,
        name=""
    ))
    fig.update_layout(
        template="plotly_dark",
        title=title,
        margin=dict(l=40, r=10, t=24, b=40),
        showlegend=False
    )
    fig.update_yaxes(title_text="dH (m)", zeroline=False,
                     gridcolor="rgba(255,255,255,0.08)")
    fig.update_xaxes(visible=False)
    return fig

def pixel_area_m2_from_ref_dem(ref_dem) -> float:
    """
    Оцінка площі пікселя в м² за референсним DEM (xdem.DEM).
    Підтримує як проєктовані CRS (в метрах), так і географічні (в градусах).
    """
    with rasterio.open(ref_dem.filename) as ds:
        tr = ds.transform
        crs = ds.crs
        px_w = abs(tr.a)      # розмір пікселя по X (в одиницях CRS)
        px_h = abs(tr.e)      # розмір пікселя по Y (в одиницях CRS)

        if crs and crs.is_projected:
            # CRS у метрах -> площа = ширина * висота
            return float(px_w * px_h)

        # Географічний CRS (градуси): оцінюємо метри/градус на широті центру
        b = ds.bounds
        lat_c = 0.5 * (b.top + b.bottom)

        # наближені формули метрів за градус довготи/широти (залежить від широти)
        # джерело: стандартні геодезичні апроксимації
        rad = np.deg2rad(lat_c)
        m_per_deg_lat = 111132.92 - 559.82*np.cos(2*rad) + 1.175*np.cos(4*rad) - 0.0023*np.cos(6*rad)
        m_per_deg_lon = 111412.84*np.cos(rad) - 93.5*np.cos(3*rad) + 0.118*np.cos(5*rad)

        w_m = px_w * m_per_deg_lon
        h_m = px_h * m_per_deg_lat
        return float(w_m * h_m)



def flood_metrics(A: np.ndarray, B: np.ndarray, px_area_m2: float) -> dict:
    """
    Бінарні маски A/B -> IoU/F1/precision/recall + площі.
    """
    A = A.astype(bool); B = B.astype(bool)
    tp = np.sum(A & B); fp = np.sum((~A) & B); fn = np.sum(A & (~B))

    denom_iou = tp + fp + fn
    iou = (tp / denom_iou) if denom_iou else np.nan

    denom_p = tp + fp
    precision = (tp / denom_p) if denom_p else np.nan

    denom_r = tp + fn
    recall = (tp / denom_r) if denom_r else np.nan

    f1 = (2 * precision * recall / (precision + recall)) \
        if (np.isfinite(precision) and np.isfinite(recall) and (precision + recall)) else np.nan

    area_A = float(np.sum(A) * px_area_m2)
    area_B = float(np.sum(B) * px_area_m2)
    d_area = abs(area_A - area_B)

    return {
        "IoU": float(iou),
        "F1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "area_A_m2": area_A,
        "area_B_m2": area_B,
        "delta_area_m2": d_area
    }

# --- Plotly: стовпчики площ затоплення і «перекриття» ---
def plotly_flood_areas_figure(stats: dict, title="Flooded area comparison"):
    df = pd.DataFrame([
        {"Scenario": "DEM 1", "Area (km²)": stats["area_A_m2"]/1e6},
        {"Scenario": "DEM 2", "Area (km²)": stats["area_B_m2"]/1e6},
        {"Scenario": "Δ |A−B|", "Area (km²)": stats["delta_area_m2"]/1e6},
    ])
    fig = go.Figure(go.Bar(x=df["Scenario"], y=df["Area (km²)"], text=[f"{v:.2f}" for v in df["Area (km²)"]], textposition="auto"))
    fig.update_layout(template="plotly_dark", title=title, yaxis_title="Area (km²)", margin=dict(l=8,r=8,t=28,b=38))
    return fig

# --- Готуємо 3-класний overlay для flood порівняння ---
def flood_compare_overlay_png(A: np.ndarray, B: np.ndarray, ref_dem, colors=((0,0,0,0), (255,0,0,180), (0,255,0,180), (255,255,0,180))):
    """
    Класи: 0=none, 1=only A, 2=only B, 3=both
    Повертає base64 PNG для BitmapLayer.
    """
    # 0 none, 1 A\B, 2 B\A, 3 A∩B
    cls = np.zeros(A.shape, dtype=np.uint8)
    onlyA = A & ~B
    onlyB = B & ~A
    both  = A & B
    cls[onlyA] = 1; cls[onlyB] = 2; cls[both] = 3

    # матриця RGBA
    pal = np.array(colors, dtype=np.uint8)  # 4x4
    rgba = pal[cls]  # HxWx4

    with rasterio.open(ref_dem.filename) as src:
        b = src.bounds
    extent = [b.left, b.right, b.bottom, b.top]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgba, extent=extent, origin="upper")
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
