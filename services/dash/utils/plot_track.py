import plotly.graph_objs as go
from utils.style import empty_dark_figure

def build_profile_figure_with_hand(
    df_all,
    df_hand,
    dem_key,
    use_hand,
    interpolated_df=None,
    interp_method=None
):
    fig = go.Figure()
    # DEM
    if not df_all.empty and f"h_{dem_key}" in df_all:
        x_axis_dem = df_all["distance_m"] if "distance_m" in df_all else df_all["x"]
        fig.add_trace(go.Scatter(
            x=x_axis_dem,
            y=df_all[f"h_{dem_key}"],
            mode="markers",
            marker=dict(size=2, color="lightgray"),
            name=f"{dem_key.upper()} DEM",
            opacity=0.9
        ))
    # ICESat-2 (raw)
    show_df = df_hand if (use_hand and not df_hand.empty) else df_all
    if not show_df.empty and "orthometric_height" in show_df:
        x_axis_ice = show_df["distance_m"] if "distance_m" in show_df else show_df["x"]
        fig.add_trace(go.Scatter(
            x=x_axis_ice,
            y=show_df["orthometric_height"],
            mode="markers",
            marker=dict(size=2, color="crimson"),
            name="ICESat-2 Orthometric Height",
            opacity=0.9
        ))

    # Interpolated profile (smooth line)
    if interpolated_df is not None and not interpolated_df.empty:
        fig.add_trace(go.Scatter(
            x=interpolated_df["distance_m"],
            y=interpolated_df["orthometric_height"],
            mode="lines",
            line=dict(width=1.5, dash="solid", color="royalblue"),
            name=f"Interpolated ({interp_method})",
            opacity=0.8
        ))

    # --- 4. Аннотація по похибці DEM
    if f"delta_{dem_key}" in df_all and not df_all[f"delta_{dem_key}"].dropna().empty:
        delta = df_all[f"delta_{dem_key}"].dropna()
        stats_text = (
            f"Похибка {dem_key.upper()}: "
            f"Сер: {delta.mean():.2f} м, "
            f"Мін: {delta.min():.2f} м, "
            f"Макс: {delta.max():.2f} м"
        )
        fig.add_annotation(
            text=stats_text,
            xref="paper", yref="paper",
            x=0.02, y=0.99,
            showarrow=False,
            font=dict(size=13, color="lightgray", family="monospace"),
            align="left",
            bordercolor="gray", borderwidth=1,
            xanchor="left"
        )

    # --- 5. Оформлення
    fig.update_layout(
        xaxis=dict(title="Відстань (м)", gridcolor="#666", gridwidth=0.6, griddash="dot", zerolinecolor="#555"),
        yaxis=dict(title="Ортометрична висота (м)", gridcolor="#666", gridwidth=0.3, griddash="dot", zerolinecolor="#555"),
        height=600,
        legend=dict(orientation="h", y=1.06, x=0.5, xanchor="center", font=dict(size=12), bgcolor='rgba(0,0,0,0)'),
        plot_bgcolor="#20232A",
        paper_bgcolor="#181818",
        font_color="#EEE",
        margin=dict(l=70, r=50, t=40, b=40)
    )
    return fig
