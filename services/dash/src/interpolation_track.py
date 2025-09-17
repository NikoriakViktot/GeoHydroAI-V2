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


def kalman_smooth(
    df,
    y_col="orthometric_height",
    transition_covariance=1e-3,     # Q, типово для сильного згладжування
    observation_covariance=3.0      # R, типово для "відсікання" викидів
):
    if df.empty:
        return df.copy()
    values = df[y_col].values

    kf = KalmanFilter(
        initial_state_mean=values[0],
        transition_matrices=[1],
        observation_matrices=[1],
        transition_covariance=transition_covariance,
        observation_covariance=observation_covariance,
    )
    state_means, _ = kf.smooth(values)
    df_result = df.copy()
    df_result["kalman_smooth"] = state_means
    return df_result
