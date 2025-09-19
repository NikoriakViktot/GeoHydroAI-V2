import dash
from dash import no_update
app = dash.get_app()

@app.callback(
    Output("url", "pathname"),
    Input("nav-home", "n_clicks"),
    Input("nav-dem-diff", "n_clicks"),
    Input("nav-dem-diff-1", "n_clicks"),
    Input("nav-icesat", "n_clicks"),
    Input("nav-best-dem", "n_clicks"),
    Input("nav-cdf", "n_clicks"),
    Input("nav-flood", "n_clicks"),
    prevent_initial_call=True
)
def navigate(*_):
    trig = dash.ctx.triggered_id
    if trig is None:
        return no_update
    mapping = {
        "nav-home": "/",
        "nav-dem-diff": "/dem-diff",
        "nav-dem-diff-1": "/dem-diff-1",
        "nav-icesat": "/icesat-map",
        "nav-best-dem": "/best-dem",
        "nav-cdf": "/cdf",
        "nav-flood": "/flood-test",
    }
    return mapping.get(trig, no_update)
