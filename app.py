# Import packages
from dash import Dash, html, dcc, Output, Input, State, ctx
import dash
import dash_bootstrap_components as dbc
from data import list_of_genres, list_of_metrics, metric_name_display


# Initialize the app - incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP]
app = Dash(
    __name__,
    use_pages=True,
    external_stylesheets=external_stylesheets,
    suppress_callback_exceptions=True
)


# App layout
app.layout = dbc.Container([
    dcc.Store(id='genre', data="__ALL__"),
    dcc.Store(id='metric', data="score"),
    dbc.Row([
        dbc.Navbar(
            dbc.Container(
                [
                    # Left side: About link
                    dbc.Nav(
                        [
                            dbc.NavbarBrand("MyAnimeList", href="/"),
                            dbc.NavItem(dbc.NavLink("About", href="/About")),
                            dbc.NavItem(dbc.NavLink("Visualization", href="/")),
                        ],
                        className="me-auto",
                    ),
                    
                    # Toggler button
                    dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
                ],
                fluid=True,
            ),
            color="dark",
            dark=True,
        )

    ]),

    dash.page_container
], fluid=True)

app.validation_layout = html.Div(
    [
        app.layout,
        dcc.Graph(id="genre-graph"),
        dcc.Graph(id="friendship-graph"),
        dcc.Dropdown(id="genre-dropdown"),
        dcc.Dropdown(id="metric-dropdown"),
    ]
)

# add callback for toggling the collapse on small screens
@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("genre", "data"),
    Output("metric", "data"),
    Output("genre-dropdown", "value"),
    Input("genre-dropdown", "value"),
    Input("metric-dropdown", "value"),
    Input("genre-graph", "clickData", allow_optional=True),
    Input("friendship-graph", "clickData", allow_optional=True),
    prevent_initial_call=True
)
def update_genre(dropdown_genre, dropdown_metric, clickData_genre, clickData_friend):
    triggered = ctx.triggered_id

    new_dropdown_genre = dropdown_genre

    if triggered == "genre-graph" and clickData_genre and clickData_genre.get("points"):
        cd = clickData_genre["points"][0].get("customdata")
        if cd:
            new_dropdown_genre = cd[0]  # genre

    if triggered == "friendship-graph" and clickData_friend and clickData_friend.get("points"):
        cd = clickData_friend["points"][0].get("customdata")
        # node_trace posiela [[n]] -> cd = ["Drama"]
        if cd and isinstance(cd, (list, tuple)) and len(cd) == 1:
            new_dropdown_genre = cd[0]

    store_genre = "__ALL__" if new_dropdown_genre == "All genres" else new_dropdown_genre

    return store_genre, dropdown_metric, new_dropdown_genre




# Run the app
if __name__ == '__main__':
    app.run(debug=True)