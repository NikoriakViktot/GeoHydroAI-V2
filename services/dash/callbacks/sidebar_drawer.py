# callbacks/sidebar_drawer.py
from dash import callback, Input, Output, State, no_update, ctx
app = dash.get_app()

@app.callback(
    Output("sidebar-wrap", "className"),
    Input("burger", "n_clicks"),           # мобільне меню
    Input("sidebar-backdrop", "n_clicks"), # клік по підкладці = закрити
    Input("collapse", "n_clicks"),         # десктопний collapse
    State("sidebar-wrap", "className"),
    prevent_initial_call=True
)
def toggle_sidebar(n_burger, n_backdrop, n_collapse, cls):
    cls = (cls or "").split()
    trig = ctx.triggered_id

    def add(c):
        if c not in cls: cls.append(c)
    def rem(c):
        if c in cls: cls.remove(c)

    if trig == "burger":         # мобільний drawer
        if "open" in cls: rem("open")
        else: add("open")
    elif trig == "sidebar-backdrop":
        rem("open")
    elif trig == "collapse":     # десктоп: повне ↔ вузьке
        if "collapsed" in cls: rem("collapsed")
        else: add("collapsed")

    return " ".join(cls) if cls else ""
