import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from pykalman import KalmanFilter

def interpolate_linear(df, x_col="distance_m", y_col="orthometric_height", grid=None):
    if grid is None:
        grid = np.linspace(df[x_col].min(), df[x_col].max(), 300)
    interp = np.interp(grid, df[x_col], df[y_col])
    return pd.DataFrame({x_col: grid, y_col: interp})

def interpolate_spline(df_ice, grid=None, s=0):
    df_ice = df_ice.drop_duplicates(subset="distance_m").dropna(subset=["distance_m", "orthometric_height"])
    df_ice = df_ice.sort_values("distance_m")
    if df_ice.shape[0] < 4:
        print("Too few points for spline")
        return None

    x = df_ice["distance_m"].values
    y = df_ice["orthometric_height"].values

    # Додатковий захист: x має бути строго зростаючим
    if np.any(np.diff(x) <= 0):
        print("Non-increasing x detected!")
        return None

    try:
        spline = UnivariateSpline(x, y, s=s)
        if grid is None:
            x_new = np.linspace(x.min(), x.max(), 500)
        else:
            x_new = grid
        y_new = spline(x_new)
        print("Spline OK")
        return pd.DataFrame({"distance_m": x_new, "orthometric_height": y_new})
    except Exception as e:
        print(f"Spline error: {e}")
        return None


# def kalman_smooth(
#     df,
#     y_col="orthometric_height",
#     transition_covariance=1e-3,     # Q, типово для сильного згладжування
#     observation_covariance=3.0      # R, типово для "відсікання" викидів
# ):
#     if df.empty:
#         return df.copy()
#     values = df[y_col].values
#
#     kf = KalmanFilter(
#         initial_state_mean=values[0],
#         transition_matrices=[1],
#         observation_matrices=[1],
#         transition_covariance=transition_covariance,
#         observation_covariance=observation_covariance,
#     )
#     state_means, _ = kf.smooth(values)
#     df_result = df.copy()
#     df_result["kalman_smooth"] = state_means
#     return df_result

def kalman_smooth(
    df: pd.DataFrame,
    y_col: str = "orthometric_height",
    x_col: str = "distance_m",
    transition_covariance: float = 1e-2,
    observation_covariance: float = 0.9,
    gap_break: float = 100.0,
    robust_premed: bool = True,
    roll_win: int = 11,
):
    if df.empty:
        out = df.copy()
        out["kalman_smooth"] = np.nan
        out["segment_id"] = np.nan
        return out

    d = df.sort_values(x_col).copy()
    y = d[y_col].to_numpy(dtype=float)
    x = d[x_col].to_numpy(dtype=float)

    # розбивка на сегменти
    gaps = np.where(np.diff(x) > gap_break)[0] + 1
    idx_splits = np.split(np.arange(len(d)), gaps)

    smoothed   = np.full_like(y, np.nan, dtype=float)
    segment_id = np.full(len(d), np.nan, dtype=float)

    for seg_no, idx in enumerate(idx_splits, start=1):
        if len(idx) == 0:
            continue
        yy = y[idx].astype(float)

        if robust_premed and len(idx) >= roll_win:
            s   = pd.Series(yy)
            med = s.rolling(roll_win, center=True, min_periods=1).median().to_numpy()
            e   = yy - med
            mad = 1.4826 * np.nanmedian(np.abs(e[np.isfinite(e)])) if np.isfinite(e).any() else 0.0
            if mad > 0:
                yy = np.where(np.abs(e) > 4 * mad, med, yy)

        if not np.isfinite(yy).any():
            continue
        seg_med = np.nanmedian(yy[np.isfinite(yy)])
        yy = np.where(np.isfinite(yy), yy, seg_med)

        dx     = np.diff(x[idx])
        med_dx = np.median(dx) if len(dx) else 1.0
        scale  = np.r_[med_dx, dx] / max(med_dx, 1e-6)
        q_step = np.clip(transition_covariance * scale, 1e-4, 1e-1).reshape(-1, 1, 1)

        kf = KalmanFilter(
            initial_state_mean=yy[0],
            transition_matrices=np.ones((len(idx), 1, 1)),
            observation_matrices=np.ones((len(idx), 1, 1)),
            transition_covariance=q_step,
            observation_covariance=observation_covariance,
        )
        state_means, _ = kf.smooth(yy)
        smoothed[idx] = state_means.ravel()
        segment_id[idx] = seg_no

    out = d.copy()
    out["kalman_smooth"] = smoothed
    out["segment_id"]    = segment_id  # ← головне
    return out
