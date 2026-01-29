import dash
from dash import html, dcc, callback, Input, Output, ctx
import dash_bootstrap_components as dbc
import plotly.express as px
from data import *
import circlify
import plotly.graph_objects as go
import networkx as nx
import numpy as np
import math

# PLEASE GUYS DONT REMOVE style={"height": ...} because it will stretch uncontrollably otherwise i dont know what this bug is but please be careful

dash.register_page(__name__, path='/')

slider_block = html.Div(
    [
        html.Div(
            "Percentile filter",
            style={
                "fontSize": "0.9rem",
                "fontWeight": "600",
                "color": 'black',
                "marginBottom": "2px",
            },
        ),
        html.Div(
            "Shows only genres and connections with friendship counts above the selected percentile.",
            style={
                "fontSize": "0.75rem",
                "color": "#6c757d",
                "marginBottom": "6px",
            },
        ),
        dcc.Slider(
            id="edge-threshold",
            min=0.10,
            max=0.90,
            step=0.05,
            value=0.50,
            marks={
                0.10: "10%",
                0.30: "30%",
                0.50: "50%",
                0.70: "70%",
                0.90: "90%",
            },
            tooltip={"placement": "bottom", "always_visible": False},
            updatemode="mouseup",
        ),
    ],
    style={"marginTop": "16px"},
)

x_focus_control = html.Div(
    [
        html.Div("Dropped proportion focus", style={"fontWeight": 600}),
        html.Div(
            "Adjust the visible X-axis range to reduce the impact of outliers.",
            style={"color": "#6c757d", "fontSize": "12px", "marginBottom": "10px"},
        ),
        html.Div(
            dcc.RangeSlider(
                id="corr-xrange-slider",
                min=0,
                max=1,
                step=0.01,
                value=[0, 1],
                allowCross=False,
                marks={},
                tooltip={"placement": "bottom", "always_visible": False},
            ),
            id="corr-xrange-slider-wrap",
        ),
    ],
    style={"padding": "6px 0 0 0"},
)

layout = html.Div(
    children=[
        # --- Filters card ---
        dbc.Card(
            dbc.CardBody(
                dbc.Row(
                    [
                        # Genre filter
                        dbc.Col(
                            dbc.Row(
                                [
                                    dbc.Col(
                                        html.Small("Genre", className="text-muted"),
                                        width="auto",
                                        className="pe-2",
                                    ),
                                    dbc.Col(
                                        dcc.Dropdown(
                                            id="genre-dropdown",
                                            options=(
                                                [{"label": "All genres", "value": "All genres"}] +
                                                [
                                                    {"label": g, "value": g}
                                                    for g in sorted(g for g in list_of_genres if g != "All genres")
                                                ]
                                            ),
                                            value="All genres",
                                            clearable=False,
                                        )
                                    ),
                                ],
                                align="center",
                                className="g-0",
                            ),
                            md=6,
                        ),

                        # Metric filter
                        dbc.Col(
                            dbc.Row(
                                [
                                    dbc.Col(
                                        html.Small("Scoring method", className="text-muted"),
                                        width="auto",
                                        className="pe-2",
                                    ),
                                    dbc.Col(
                                        dcc.Dropdown(
                                            id="metric-dropdown",
                                            options=[
                                                {"label": metric_name_display[m], "value": m}
                                                for m in list_of_metrics
                                            ],
                                            value="score",
                                            clearable=False,
                                        )
                                    ),
                                ],
                                align="center",
                                className="g-0",
                            ),
                            md=6,
                        ),
                    ],
                    className="g-3",
                )
            ),
            className="mb-4 shadow-sm",
        ),

        # --- Main content ---
        html.Div(
            children=[
                # Row 1: friendship + right-side charts
                dbc.Row(
                    [
                        # Left: Friendship network + slider
                        dbc.Col(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            html.H4("Friendships between users by their favorite genre"),
                                            width="auto",
                                        ),
                                        dbc.Col(
                                            html.I(
                                                className="bi bi-info-circle-fill",
                                                id="friendship-info",
                                            ),
                                            width="auto",
                                            align="center",
                                        ),
                                        dbc.Tooltip(
                                            "This network shows how users are connected based on their favorite anime genres. "
                                            "Node size represents friendships within the same genre, while edges show friendships "
                                            "between different genres. Thicker edges indicate stronger cross-genre connections. "
                                            "Use the slider and genre selection to explore stronger or weaker relationships.",
                                            target="friendship-info",
                                            placement="left",
                                        ),
                                    ],
                                    justify="between",
                                ),

                                dcc.Graph(
                                    figure={},
                                    id="friendship-graph",
                                    style={"height": "700px"},
                                ),
                                slider_block
                            ],
                            width=12,
                            md=6,
                        ),

                        # Right: Genre popularity + Top anime bar
                        dbc.Col(
                            [
                                # --- Genre popularity header ---
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            html.H4(
                                                id="title-genres",
                                                children="Popularity of genres",
                                            ),
                                            width="auto",
                                        ),
                                        dbc.Col(
                                            html.I(
                                                className="bi bi-info-circle-fill",
                                                id="genre-info",
                                                style={"cursor": "pointer"},
                                            ),
                                            width="auto",
                                            className="ms-auto",
                                        ),
                                        dbc.Tooltip(
                                            "This chart shows anime genres ranked by the selected metric. "
                                            "Bubble size represents the relative value of the metric for each genre. "
                                            "Select a genre to highlight it and explore its role across the visualizations.",
                                            target="genre-info",
                                            placement="left",
                                        ),
                                    ],
                                    align="center",
                                    className="g-2",
                                ),
                                dcc.Graph(id="genre-graph", style={"height": "350px"}),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            html.H4(
                                                id="title-barchart",
                                                children="Popularity of anime",
                                            ),
                                            width="auto",
                                        ),
                                        dbc.Col(
                                            html.I(
                                                className="bi bi-info-circle-fill",
                                                id="barchart-info",
                                                style={"cursor": "pointer"},
                                            ),
                                            width="auto",
                                            className="ms-auto",
                                        ),
                                        dbc.Tooltip(
                                            "This barchart shows the top 20 anime in the selected genre based on the selected metric. Each bar represents an anime and the y-axis shows the value of that anime based on selected metric (i.e. average rating score). When you hover a bar you see detail about the anime.",
                                            target="barchart-info",
                                            placement="left",
                                        ),
                                    ],
                                    align="center",
                                    className="g-2 mt-3",
                                ),
                                html.Div(
                                    [
                                        dcc.Graph(
                                            id="barchart", style={"height": "350px"},
                                            clear_on_unhover=True,
                                        ),
                                        dcc.Tooltip(
                                            id="barchart-tooltip",
                                            direction="right",
                                            style={
                                                "backgroundColor": "white",
                                                "padding": "10px",
                                                "zIndex": 9999,
                                                "pointerEvents": "none",
                                            },
                                        ),
                                    ],
                                    style={"position": "relative"},
                                ),
                            ],
                            width=12,
                            md=6,
                            style={"overflow": "visible"},
                        ),
                    ],
                    className="g-4 mt-4",
                ),

                # Row 2: heatmap + correlation
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            html.H4(
                                                "Average anime score by episode count and airing duration"
                                            ),
                                            width="auto",
                                        ),
                                        dbc.Col(
                                            html.I(
                                                className="bi bi-info-circle-fill",
                                                id="duration-info",
                                            ),
                                            width="auto",
                                            align="center",
                                        ),
                                        dbc.Tooltip(
                                            "This heatmap shows how average anime scores vary by airing duration and number of episodes. "
                                            "Each cell represents the average score for anime in the corresponding category, with darker "
                                            "colors indicating higher popularity.",
                                            target="duration-info",
                                            placement="left",
                                        ),
                                    ],
                                    justify="between",
                                ),
                                dcc.Graph(figure={}, id="duration", style={"height": "400px"}),
                            ],
                            width=12,
                            md=6,
                        ),
                        dbc.Col(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            html.H4(
                                                id="title-correlation",
                                                children="Relationship between popularity and dropout rate",
                                            ),
                                            width="auto",
                                        ),
                                        dbc.Col(
                                            html.I(
                                                className="bi bi-info-circle-fill",
                                                id="correlation-info",
                                            ),
                                            width="auto",
                                            align="center",
                                        ),
                                        dbc.Tooltip(
                                            "This graph shows the relationship between the selected popularity metric "
                                            "and the dropout rate (proportion of users who dropped the anime after starting it). "
                                            "Use this graph to explore how different genres perform in terms of user retention "
                                            "and popularity. A dropout rate greater than 1 indicates that more users dropped "
                                            "the anime than completed it!",
                                            target="correlation-info",
                                            placement="left",
                                        ),
                                    ],
                                    justify="between",
                                ),
                                dcc.Graph(
                                    figure={},
                                    id="correlation",
                                    style={"height": "400px"},
                                    clear_on_unhover=True,
                                ),
                                dcc.Tooltip(
                                    id="correlation-tooltip",
                                    direction="right",
                                    style={
                                        "backgroundColor": "white",
                                        "padding": "10px",
                                        "zIndex": 9999,
                                        "pointerEvents": "none",
                                    },
                                ),
                                x_focus_control
                            ],
                            width=12,
                            md=6,
                        ),
                    ],
                        className="g-4 mt-4",
                ),
            ],
            style={
                "paddingLeft": "min(1cm, 4vw)",
                "paddingRight": "min(1cm, 4vw)",
            }
        ),
    ]
)

@callback(
    Output("title-genres", "children"),
    Output("title-barchart", "children"),
    Output("title-correlation", "children"),
    Input("genre-dropdown", "value"),
    Input("metric-dropdown", "value"),
)
def update_titles(genre_new, metric_new):
    metric_txt = metric_name_display.get(metric_new, str(metric_new))

    if genre_new in (None, "__ALL__", "All genres"):
        genre_txt = "All genres"
        return (
            f"Popularity of genres by {metric_txt}",
            "Top 20 anime",
            f"{metric_txt} vs Dropout rate – {genre_txt}",
        )

    genre_txt = genre_new
    return (
        f"Popularity of genres by {metric_txt}",
        f"Top 20 anime in {genre_txt} by {metric_txt}",
        f"{metric_txt} vs Dropout rate – {genre_txt}",
    )


@callback(
    Output("corr-xrange-slider", "min"),
    Output("corr-xrange-slider", "max"),
    Output("corr-xrange-slider", "step"),
    Output("corr-xrange-slider", "value"),
    Output("corr-xrange-slider", "marks"),
    Input("genre-dropdown", "value"),
)
def update_corr_xrange_slider(genre_new):

    if genre_new in (None, "All genres", "__ALL__"):
        df = correlation_details_df.copy()
    else:
        df = correlation_details_df[correlation_details_df["genre"] == genre_new].copy()

    df["completed_count"] = pd.to_numeric(df["completed_count"], errors="coerce")
    df["dropped_count"] = pd.to_numeric(df["dropped_count"], errors="coerce")
    df = df[df["completed_count"] > 0].copy()

    df["dropped_proportion"] = df["dropped_count"] / df["completed_count"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["dropped_proportion"])

    x_min = -0.25
    step = 0.25

    if df.empty:
        x_max = 1
        value = [x_min, x_max]
        marks = {x_min: "-0.25", 0: "0", 1: "1"}
        return x_min, x_max, step, value, marks

    real_max = float(df["dropped_proportion"].max())
    x_max = max(1, math.ceil(real_max))

    value = [x_min, x_max]

    marks = {x_min: "-0.25"}
    for i in range(0, x_max + 1):
        marks[i] = str(i)

    return x_min, x_max, step, value, marks

# Update friendship graph on genre or metric change
# TODO change component id for output to correct one
@callback(
    Output(component_id='friendship-graph', component_property='figure'),
    Input(component_id='genre', component_property='data'),
    Input(component_id='metric', component_property='data'),
    Input(component_id="edge-threshold", component_property="value"),

)
def update_friend_graph(genre_new, metric_new, threshold):
    threshold_val =  friendship_df["friendship_count"].quantile(threshold)
    within_genre_leaderboard = create_friendship_within_leaderboard(friendship_df, threshold_val)
    genres_list = within_genre_leaderboard['genre'].to_list()
    between_genre_edges = create_friendship_between_counts(friendship_df, set(genres_list), threshold_val)

    fig = build_friendship_network_figure(
        within_df=within_genre_leaderboard,
        between_df=between_genre_edges,
        genre=genre_new,
        seed=7,
        k=0.9,
    )

    return fig

def build_friendship_network_figure(within_df, between_df, genre=None, seed=7, k=1.4, iterations=200):
    G = nx.Graph()

    # Nodes
    for _, row in within_df.iterrows():
        G.add_node(
            row["genre"],
            same_count=float(row["friendship_count"])
        )

    # Edges
    for _, row in between_df.iterrows():
        g1, g2 = row["genre_1"], row["genre_2"]
        w = float(row["friendship_count"])

        if g1 not in G:
            G.add_node(g1, same_count=0.0)
        if g2 not in G:
            G.add_node(g2, same_count=0.0)

        G.add_edge(g1, g2, weight=w)

    if G.number_of_nodes() == 0:
        return go.Figure().update_layout(title="Friendships among users")

    pos = nx.spring_layout(
        G,
        seed=seed,
        k=k,
        iterations=iterations,
        weight=None,
    )

    nodes = list(G.nodes())

    has_selection = (
        genre is not None
        and genre not in {"__ALL__", "All genres"}
        and genre in nodes
    )
    selected = genre if has_selection else None

    # ---------- EDGES (UNCHANGED) ----------
    edge_traces = []

    weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(weights) if weights else 1.0
    max_w = max(max_w, 1.0)

    hit_x, hit_y, hit_text = [], [], []
    HOVER_POINTS_PER_EDGE = 8

    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        w = G[u][v]["weight"]

        width = 1 + 6 * np.log1p(w) / np.log1p(max_w)
        is_highlight = has_selection and (u == selected or v == selected)

        if not has_selection:
            line_color = "rgba(60, 60, 60, 0.45)"
            line_opacity = 0.65
        else:
            if is_highlight:
                line_color = "rgba(60, 60, 60, 0.55)"
                line_opacity = 0.75
            else:
                line_color = "rgba(60, 60, 60, 0.35)"
                line_opacity = 0.45

        edge_traces.append(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(color=line_color, width=width),
                opacity=line_opacity,
                hoverinfo="skip",
                showlegend=False,
            )
        )

        for t in np.linspace(0.12, 0.88, HOVER_POINTS_PER_EDGE):
            hit_x.append(x0 * (1 - t) + x1 * t)
            hit_y.append(y0 * (1 - t) + y1 * t)
            hit_text.append(
                f"<b>{u} ↔ {v}</b><br><b>Between-genre Friendships: </b>{int(w)}"
            )

    edge_hover_trace = go.Scatter(
        x=hit_x,
        y=hit_y,
        mode="markers",
        marker=dict(size=18, opacity=0),
        hoverinfo="text",
        text=hit_text,
        showlegend=False,
        name="edge_hover",
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="#cccccc",
            font=dict(color="#222222", size=13),
            align="left",
            namelength=-1
        ),
    )

    # ---------- NODES ----------
    same_counts = np.array([G.nodes[n]["same_count"] for n in nodes], dtype=float)

    max_same = max(same_counts) if len(same_counts) else 1.0
    max_same = max(max_same, 1.0)

    node_sizes = [max(18, 22 + 65 * s / max_same) for s in same_counts]

    node_opacity = []
    node_line_width = []
    node_line_color = []

    for n in nodes:
        if not has_selection:
            node_opacity.append(1.0)
            node_line_width.append(1)
            node_line_color.append("rgba(0,0,0,0.25)")
        else:
            if n == selected:
                node_opacity.append(1.0)
                node_line_width.append(2)
                node_line_color.append("#1f2d3d")
            else:
                node_opacity.append(0.40)
                node_line_width.append(1)
                node_line_color.append("rgba(0,0,0,0.15)")

    node_trace = go.Scatter(
        x=[pos[n][0] for n in nodes],
        y=[pos[n][1] for n in nodes],
        mode="markers+text",
        text=nodes,
        textposition="middle center",
        textfont=dict(size=12),
        customdata=[[n] for n in nodes],
        name="friend_nodes",
        hoverinfo="text",
        hovertext=[
            f"<b>{n}</b><br><b>Within-genre Friendships</b>: {int(G.nodes[n]['same_count'])}"
            for n in nodes
        ],
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="#cccccc",
            font=dict(color="#222222", size=13),
            align="left",
            namelength=-1
        ),
        marker=dict(
            size=node_sizes,
            line=dict(width=node_line_width, color=node_line_color),
            color=[genre_color[n] for n in nodes],
            opacity=node_opacity,
        ),
        showlegend=False,
    )

    fig = go.Figure(data=edge_traces + [edge_hover_trace, node_trace])

    fig.update_layout(
        title=None,
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        hovermode="closest",
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    return fig



def create_friendship_within_leaderboard(friendship_df, threshold):
    within_df = friendship_df[
        friendship_df["genre_1"] == friendship_df["genre_2"]
    ].copy()

    within_leaderboard = (
        within_df[["genre_1", "friendship_count"]]
        .rename(columns={"genre_1": "genre"})
        .loc[lambda df: df["friendship_count"] >= threshold]
        .sort_values("friendship_count", ascending=False)
        .reset_index(drop=True)
    )

    return within_leaderboard


def create_friendship_between_counts(friendship_df, genres, threshold):
    between_edges = friendship_df[
        (friendship_df["genre_1"] != friendship_df["genre_2"])
        & friendship_df["genre_1"].isin(genres)
        & friendship_df["genre_2"].isin(genres)
        & (friendship_df["friendship_count"] >= threshold)
    ][["genre_1", "genre_2", "friendship_count"]].reset_index(drop=True)

    return between_edges


# Update genre graph on genre or metric change
@callback(
    Output(component_id='genre-graph', component_property='figure'),
    Input(component_id='genre', component_property='data'),
    Input(component_id='metric', component_property='data'),
)
def update_genre_graph(genre_new, metric_new):
    if metric_new == "completed_count":
        metric_orig = "avg_completed_count"
    elif metric_new == "score":
        metric_orig = "avg_score"
    elif metric_new == "favorites_count":
        metric_orig = "avg_favorites_count"

    data = top_m_genres_by_num.copy()
    data = data[["genre", metric_orig]].rename(columns={metric_orig: metric_new})

    fig = genre_bubble_chart(
        data,
        metric_label=metric_new,
        chosen_genre=genre_new,
        cols=6,
        title_prefix=f"Genre popularity by"
    )

    return fig


def genre_bubble_chart(df, metric_label, chosen_genre, cols=6, title_prefix="Genres by"):

    df[metric_label] = pd.to_numeric(df[metric_label], errors="coerce").fillna(0)
    df = df.sort_values(metric_label, ascending=False).reset_index(drop=True)

    items = [{"id": genre, "datum": float(val)} for genre, val in zip(df["genre"], df[metric_label])]

    circles = circlify.circlify(
        items,
        show_enclosure=False,
        target_enclosure=circlify.Circle(x=0, y=0, r=1.0),
    )

    # circlify returns circles in random order, so we have to handle it while adding labels later
    leaf = [c for c in circles if c.level == 1]

    # bubbles placed in oval shape not in circle
    stretch_x = 1.75
    stretch_y = 1.0

    xs = [c.x * stretch_x for c in leaf]
    ys = [c.y * stretch_y for c in leaf]
    rs = [c.r for c in leaf]

    gap = 0.005 # some space around each bubble

    value_map = dict(zip(df["genre"], df[metric_label].astype(float)))
    genres_leaf = [c.ex["id"] if isinstance(c.ex, dict) else c.ex for c in leaf]
    values_leaf = [value_map.get(g, 0.0) for g in genres_leaf]

    shapes = []
    for (g, x, y, r) in zip(genres_leaf, xs, ys, rs):
        has_selection = (chosen_genre is not None and chosen_genre != "__ALL__")
        is_selected = has_selection and (g == chosen_genre)
        fill_alpha = 1.0 if (not has_selection or is_selected) else 0.5

        r = max(r - gap, 0)
        shapes.append(dict(
            type="circle",
            xref="x", yref="y",
            x0=x - r, x1=x + r,
            y0=y - r, y1=y + r,
            line=dict(
                width=2 if is_selected else 1, 
                color="#1f2d3d" if is_selected else "rgba(0,0,0,0.25)"
            ),
            fillcolor=_hex_to_rgba(genre_color[g], fill_alpha),
            layer="below"
        ))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=xs,
        y=ys,
        mode="markers+text",
        text=[g if r >= 0.085 else "" for g, r in zip(genres_leaf, rs)],
        textposition="middle center",
        customdata=list(zip(genres_leaf, values_leaf)),
        hovertemplate=f"<b>%{{customdata[0]}}</b><br><b>{metric_name_display[metric_label]}:</b> %{{customdata[1]}}<extra></extra>",
        marker=dict(
            size=[max(8, r * 260) for r in rs],
            sizemode="diameter",
            opacity=0.01,
            line=dict(width=0),
        ),
        textfont=dict(size=12),
    ))

    fig.update_layout(
        title=None,
        shapes=shapes,
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis=dict(visible=False, range=[-1.05, 1.05]),
        yaxis=dict(visible=False, range=[-1.05, 1.05]),
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="white",
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="#cccccc",
            font=dict(color="#222222", size=13),
            align="left",
            namelength=-1
        ),
        clickmode="event+select",
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    return fig

def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"

# Update top anime barchart on genre or metric change
@callback(
    Output(component_id='barchart', component_property='figure'),
    Input(component_id='genre', component_property='data'),
    Input(component_id='metric', component_property='data'),
)
def update_barchart(genre_new, metric_new):
    metric = metric_new
    metric_df_top_anime = top_anime_details_df[(top_anime_details_df["genre"] == genre_new) & (top_anime_details_df["metric"] == metric)].sort_values(by="rank")

    fig = px.bar(
        metric_df_top_anime, 
        x="title", 
        y=metric, 
        color_discrete_sequence=[genre_color[genre_new]],
        hover_data=metric_df_top_anime.columns,
        title=None
    )
    
    # Customize x-axis to show truncated labels
    fig.update_xaxes(
        tickangle=45,
        tickmode='array',
        tickvals=metric_df_top_anime["title"],
        ticktext=metric_df_top_anime["title_truncated"],
        title="Title"
    )

    fig.update_yaxes(
        title=metric_name_display[metric]
    )
    
    # Optional: Improve hover info
    fig.update_traces(
        hoverinfo="none",
        hovertemplate=None,
        customdata=metric_df_top_anime[['title', 'rank', 'main_pic', metric]].assign(metric_name=metric).values
    )
    
    fig.update_layout(
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="#cccccc",
            font=dict(color="#222222", size=13),
            align="left",
            namelength=-1
        )
    )

    return fig

@callback(
    Output("barchart-tooltip", "show"),
    Output("barchart-tooltip", "bbox"),
    Output("barchart-tooltip", "children"),
    Input("barchart", "hoverData"),
)
def display_barchart_tooltip(hoverData):
    if hoverData is None:
        return False, dash.no_update, dash.no_update

    point = hoverData["points"][0]
    title, rank, main_pic, metric_val, metric_name = point["customdata"]
    bbox = point["bbox"]  # exact bar position

    children = html.Div(
        [
            html.Img(
                src=main_pic,
                style={
                    "width": "120px",
                    "display": "block",
                    "margin": "0 auto 8px auto",
                },
            ),
            html.B(title),
            html.Br(),
            html.Span([html.B("Rank: "), rank]),
            html.Br(),
            html.Span([html.B(f"{metric_name_display[metric_name]}: "), metric_val]),
        ]
    )

    return True, bbox, children

# show details of the clicked element
@callback(
    Output(component_id='details', component_property='children'),
    Output(component_id="hide-detail", component_property="style"),
    Input(component_id='barchart', component_property='clickData'),
    Input(component_id='hide-detail', component_property='n_clicks')
)
def show_detail_click(clickData, n_clicks):
    # if this was triggered by barchart
    if ctx.triggered_id == 'barchart':
        title, rank, main_pic, metric_val, metric_name = clickData['points'][0]['customdata']

        row_children = [
            dbc.Col(
                dbc.CardImg(
                    src=main_pic,
                    className="img-fluid rounded-start",
                ),
                className="col-md-4",
            ),
            dbc.Col(
                dbc.CardBody(
                    [
                        html.H4(title, className="card-title"),
                        dcc.Markdown(f'''
                            **Title:** {title}

                            **{metric_name.title()}:** {metric_val}

                            **Rank:** {rank}
                        '''),
                        html.Small(
                            "Last updated 3 mins ago",
                            className="card-text text-muted",
                        ),
                    ]
                ), className="col-md-8",
            ),
        ]

        show_button_style = style={"display": "block", "background-color": MYANIMELIST_COLOR, "background-image": None}
        return row_children, show_button_style
    
    if ctx.triggered_id == 'hide-detail':
        return [], {"display": "none"}
    
    return [], {"display": "none"}

# Update duration vs num episode graph on genre or metric change
# TODO change component id for output to correct one
@callback(
    Output(component_id='duration', component_property='figure'),
    Input(component_id='genre', component_property='data'),
    Input(component_id='metric', component_property='data'),
)
def update_duration_graph(genre_new, metric_new):
    z = pivot_avg.values

    custom = np.dstack([pivot_avg.values, pivot_cnt.values])

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=EP_LABELS,
            y=WK_LABELS,
            customdata=custom,
            colorscale=colorscale_heatmap,
            xgap=2,
            ygap=2,
            hovertemplate=(
                "<b>Episodes</b>: %{x}<br>"
                "<b>Airing Weeks</b>: %{y}<br>"
                "<b>Average Score</b>: %{customdata[0]:.2f}<br>"
                "<b>Anime Count</b>: %{customdata[1]:.0f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=None,
        xaxis_title="Number of episodes",
        yaxis_title="Airing duration (weeks)",
        margin=dict(l=60, r=30, t=60, b=60),
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="#cccccc",
            font=dict(color="#222222", size=13),
            align="left",
            namelength=-1
        )
    )

    fig.update_xaxes(type="category")
    fig.update_yaxes(type="category")

    return fig

@callback(
    Output("correlation", "figure"),
    Input("genre-dropdown", "value"),
    Input("metric-dropdown", "value"),
    Input("corr-xrange-slider", "value"),
)
def update_correlation_graph(genre_new, metric_new, x_range):
    metric = metric_new
    metric_display = metric_name_display[metric]

    if genre_new in (None, "All genres", "__ALL__"):
        df = correlation_details_df.copy()
    else:
        df = correlation_details_df[correlation_details_df["genre"] == genre_new].copy()

    df["completed_count"] = pd.to_numeric(df["completed_count"], errors="coerce")
    df["dropped_count"] = pd.to_numeric(df["dropped_count"], errors="coerce")
    df[metric] = pd.to_numeric(df[metric], errors="coerce")

    df = df[df["completed_count"] > 0].copy()
    df["dropped_proportion"] = df["dropped_count"] / df["completed_count"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["dropped_proportion", metric])

    color = genre_color.get(genre_new, MYANIMELIST_COLOR)

    fig = px.scatter(
        df,
        x="dropped_proportion",
        y=metric,
        color_discrete_sequence=[color],
        title=None,
    )

    fig.update_traces(
        hoverinfo="none",
        hovertemplate=None,
        customdata=df[["title", "dropped_count", "completed_count", "main_pic", "dropped_proportion"]].values,
    )

    if x_range and len(x_range) == 2:
        x0, x1 = x_range
        x0 = max(-0.5, x0)
        if x1 <= x0:
            x1 = x0 + 1
        fig.update_xaxes(range=[x0, x1])
    else:
        fig.update_xaxes(autorange=True)

    fig.update_xaxes(title="Dropped Proportion (Dropped / Completed)")
    fig.update_yaxes(title=metric_display)

    return fig

@callback(
    Output("correlation-tooltip", "show"),
    Output("correlation-tooltip", "bbox"),
    Output("correlation-tooltip", "children"),
    Input("correlation", "hoverData"),
)
def display_correlation_tooltip(hoverData):
    if hoverData is None:
        return False, None, None

    point = hoverData["points"][0]
    title, dropped_count, completed_count, main_pic, dropped_proportion = point["customdata"]
    bbox = point["bbox"]

    children = html.Div(
        [
            html.Img(
                src=main_pic,
                style={
                    "width": "120px",
                    "display": "block",
                    "margin": "0 auto 8px auto",
                },
            ),
            html.Span([html.B("Title: "), title]),
            html.Br(),
            html.Span([html.B("Dropped Proportion: "), round(dropped_proportion, 2)]),
            html.Br(),
            html.Span([html.B("Dropped Count: "), dropped_count]),
            html.Br(),
            html.Span([html.B("Completed Count: "), completed_count]),
        ]
    )

    return True, bbox, children
