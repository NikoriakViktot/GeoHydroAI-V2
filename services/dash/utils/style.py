#utils/style.py
import plotly.graph_objs as go


sidebar_style = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "280px",
    "padding": "20px",
    "backgroundColor": "#1c1c1c",
    "color": "#EEEEEE",
    "overflowY": "auto",
    "borderRight": "1px solid #333",
    "zIndex": 1000
}

label_style = {
    "marginTop": "12px",
    "marginBottom": "4px",
    "fontWeight": "bold",
    "fontSize": "14px"
}

dropdown_style = {
    "backgroundColor": "#23272b",
    "color": "#EEEEEE",
    "borderRadius": "6px",
    "border": "1px solid #444",
    "marginBottom": "10px"
}

button_style = {
    "backgroundColor": "#2d8cff",
    "color": "#fff",
    "padding": "8px 18px",
    "border": "none",
    "borderRadius": "6px",
    "fontWeight": "bold",
    "cursor": "pointer",
    "marginTop": "14px"
}
dark_table_style = dict(
    style_table={
        "backgroundColor": "#181818",
        "border": "none",
        "borderRadius": "8px",
        "overflowX": "auto",
    },
    style_header={
        "backgroundColor": "#23272b",
        "color": "#EEEEEE",
        "fontWeight": "bold",
        "fontSize": "15px",
        "border": "none",
    },
    style_cell={
        "backgroundColor": "#23272b",
        "color": "#EEEEEE",
        "border": "1px solid #333",
        "fontFamily": "Segoe UI, Verdana, Arial, sans-serif",
        "fontSize": "14px",
        "textAlign": "center",
        "padding": "7px 5px",
        "whiteSpace": "normal",
    },
    style_data_conditional=[
        {
            "if": {"row_index": 0},
            "backgroundColor": "#323b32",
            "color": "#d4edda",
            "fontWeight": "bold",
        },
        {
            "if": {"state": "selected"},
            "backgroundColor": "#114488",
            "color": "#FFF",
        },
        {
            "if": {"row_index": "odd"},
            "backgroundColor": "#22282e",
        },
    ],
)


tab_style = {
    "background": "#23272b",
    "color": "#bdbdbd",
    "border": "none",
    "padding": "12px 22px",
    "fontWeight": "bold",
    "fontSize": "17px",
    "marginRight": "2px",
    "borderRadius": "10px 10px 0 0",
    "transition": "background 0.3s, color 0.3s",
    "boxShadow": "0 1px 6px #0001",
}

selected_tab_style = {
    "background": "#2d8cff",
    "color": "#fff",
    "border": "none",
    "padding": "12px 22px",
    "fontWeight": "bold",
    "fontSize": "18px",
    "borderRadius": "12px 12px 0 0",
    "boxShadow": "0 4px 12px #0018",
    "transition": "background 0.3s, color 0.3s",
    "outline": "none",
}
dark_card_style = {
    "marginTop": "20px",
    "fontWeight": "bold",
    "backgroundColor": "#23272b",
    "color": "#B9E0FF",
    "borderRadius": "14px",
    "padding": "14px 22px",
    "fontFamily": "monospace",
    "fontSize": "18px",
    "letterSpacing": "0.03em",
    "border": "1.5px solid #344",
    "boxShadow": "0 2px 10px #10161a70",
    "display": "inline-block",
    "maxWidth": "97%"
}

def apply_dark_theme(fig):
    fig.update_layout(
        paper_bgcolor="#181818",
        plot_bgcolor="#181818",
        font_color="#EEEEEE",
        xaxis=dict(color="#EEEEEE"),
        yaxis=dict(color="#EEEEEE"),
        legend=dict(font_color="#EEEEEE"),
    )
    return fig

dark_table_style = dict(
    style_table={"backgroundColor": "#222", 'overflowX': 'auto', "marginTop": "10px"},
    style_header={
        "backgroundColor": "#181818",
        "color": "#EEEEEE",
        "fontWeight": "bold",
        "fontSize": "16px",
        "border": "1px solid #333"
    },
    style_cell={
        "backgroundColor": "#222",
        "color": "#EEEEEE",
        "border": "1px solid #333",
        "fontFamily": "Segoe UI, Verdana, Arial, sans-serif",
        "fontSize": "15px",
        "textAlign": "center",
        "padding": "5px",
    },
    style_data_conditional=[
        {
            "if": {"row_index": 0},
            "backgroundColor": "#323b32",
            "color": "#d4edda",
            "fontWeight": "bold",
        }
    ],
)

dark_card_style = {
    "marginTop": "20px",
    "fontWeight": "bold",
    "backgroundColor": "#23272b",
    "color": "#B9E0FF",
    "borderRadius": "14px",
    "padding": "14px 22px",
    "fontFamily": "monospace",
    "fontSize": "18px",
    "letterSpacing": "0.03em",
    "border": "1.5px solid #344",
    "boxShadow": "0 2px 10px #10161a70",
}

dark_dropdown_style = {
    "backgroundColor": "#23272b",
    "color": "#EEEEEE",
    "border": "1px solid #444",
    "borderRadius": "8px",
}


def apply_dark_theme(fig):
    fig.update_layout(
        paper_bgcolor="#181818",   # Фон "листа" (навколо графіка)
        plot_bgcolor="#181818",    # Фон самої області графіка
        font_color="#EEEEEE",      # Колір усіх підписів
        xaxis=dict(
            color="#EEEEEE",
            gridcolor="#333",      # Колір ліній сітки по X
            zerolinecolor="#555",  # Колір осі X
        ),
        yaxis=dict(
            color="#EEEEEE",
            gridcolor="#333",
            zerolinecolor="#555",
        ),
        legend=dict(
            font_color="#EEEEEE",
            bgcolor="#23272b"
        ),
        margin=dict(l=45, r=30, t=50, b=40),  # Додай трохи місця
    )
    return fig

def empty_dark_figure(height=600, text=None):
    fig = go.Figure()
    fig.update_layout(
        plot_bgcolor="#20232A",
        paper_bgcolor="#181818",
        font_color="#EEE",
        height=height,
        xaxis=dict(visible=False, showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(visible=False, showticklabels=False, showgrid=False, zeroline=False),
        margin=dict(l=70, r=50, t=40, b=40)
    )
    if text:
        fig.add_annotation(
            text=text,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="#AAA", family="monospace"),
            xanchor="center", yanchor="middle"
        )
    return fig
