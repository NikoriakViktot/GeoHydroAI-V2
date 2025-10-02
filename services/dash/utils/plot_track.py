import plotly.graph_objs as go
from utils.style import empty_dark_figure
import pandas as pd
import numpy as np



def build_profile_figure_with_hand(
    df_all, df_hand, dem_key, use_hand,
    interpolated_df=None, interp_method=None,
    q_label=None, r_label=None
):
    ...
    # --- сортування по відстані (щоб gap-break працював коректно в лінії)
    for _df in (df_all, df_hand, interpolated_df):
        if isinstance(_df, pd.DataFrame) and not _df.empty and "distance_m" in _df:
            _df.sort_values("distance_m", inplace=True)

    fig = go.Figure()

    # DEM (фон)
    if not df_all.empty and f"h_{dem_key}" in df_all:
        x_axis_dem = df_all["distance_m"] if "distance_m" in df_all else df_all["x"]
        fig.add_trace(go.Scattergl(
            x=x_axis_dem, y=df_all[f"h_{dem_key}"],
            mode="markers", marker=dict(size=2), name=f"{dem_key.upper()} DEM",
            opacity=0.9, marker_color="lightgray"
        ))

    # ICESat-2 (raw або HAND-фільтрований)
    show_df = df_hand if (use_hand and not df_hand.empty) else df_all
    if not show_df.empty and "orthometric_height" in show_df:
        x_axis_ice = show_df["distance_m"] if "distance_m" in show_df else show_df["x"]
        fig.add_trace(go.Scattergl(
            x=x_axis_ice, y=show_df["orthometric_height"],
            mode="markers", marker=dict(size=2), name="ICESat-2 Orthometric Height",
            opacity=0.9, marker_color="crimson"
        ))

    # Калман/інтерпольована лінія (поважає gap-break через NaN)
    if isinstance(interpolated_df, pd.DataFrame) and not interpolated_df.empty:
        fig.add_trace(go.Scatter(
            x=interpolated_df["distance_m"],
            y=interpolated_df["orthometric_height"],
            mode="lines",
            line=dict(width=1.6, dash="solid"),
            name=("Interpolated ("
                  + (interp_method or "smooth")
                  + (f", Q={q_label}" if q_label is not None else "")
                  + (f", R={r_label}" if r_label is not None else "")
                  + ")"),
            opacity=0.9,
            connectgaps=False  # ← важливо: не зшивати прогалини
        ))

    # Метрики похибки: якщо є калман-профіль -> delta_*_kalman, інакше delta_*
    err_text = None
    if not df_all.empty:
        err_col_kal = f"delta_{dem_key}_kalman"
        err_col_raw = f"delta_{dem_key}"
        if interpolated_df is not None and err_col_kal in df_all:
            delta = df_all[err_col_kal].dropna()
            tag = f"{dem_key.upper()} vs Kalman"
        elif err_col_raw in df_all:
            delta = df_all[err_col_raw].dropna()
            tag = f"{dem_key.upper()}"
        else:
            # якщо delta_* колонок нема — порахувати на льоту з того, що є
            base = df_all.get(f"h_{dem_key}")
            ref  = interpolated_df["orthometric_height"] if (interpolated_df is not None and
                                                             "orthometric_height" in interpolated_df) \
                   else show_df.get("orthometric_height")
            if base is not None and ref is not None:
                tmp = pd.Series(base.values - np.interp(
                    df_all["distance_m"].values,
                    (interpolated_df or show_df)["distance_m"].values,
                    (interpolated_df or show_df)["orthometric_height"].values,
                    left=np.nan, right=np.nan
                ))
                delta = tmp.dropna()
                tag = f"{dem_key.upper()} (on-the-fly)"
            else:
                delta = pd.Series(dtype=float); tag = None

        if not delta.empty and tag:
            err_text = (
                f"Error {tag}: "
                f"Mean: {delta.mean():.2f} m, "
                f"Min: {delta.min():.2f} m, "
                f"Max: {delta.max():.2f} m"
            )
    if err_text:
        fig.add_annotation(
            text=err_text, xref="paper", yref="paper", x=0.02, y=0.99,
            showarrow=False, font=dict(size=13, color="lightgray", family="monospace"),
            align="left", bordercolor="gray", borderwidth=1, xanchor="left"
        )

    fig.update_layout(
        xaxis=dict(
            title="Distance along track (m)",
            gridcolor="#666", gridwidth=0.6, griddash="dot",
            zerolinecolor="#555"
        ),
        yaxis=dict(
            title="Orthometric elevation (m)",
            gridcolor="#666", gridwidth=0.3, griddash="dot",
            zerolinecolor="#555"
        ),
        height=600,
        legend=dict(
            title="Data sources",
            orientation="h", y=1.06, x=0.5, xanchor="center",
            font=dict(size=12),
            bgcolor="rgba(0,0,0,0)"
        ),
        plot_bgcolor="#20232A",
        paper_bgcolor="#181818",
        font_color="#EEE",
        margin=dict(l=70, r=50, t=40, b=40)
    )
    return fig
