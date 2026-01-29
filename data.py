import pandas as pd
import numpy as np

MAX_NUM_OF_GENRES = 20
MYANIMELIST_COLOR = '#2e51a2'

# Incorporate data
genre_top20_stats_df = pd.read_csv("data/genre_popularity_top20_stats.csv")
top_anime_df = pd.read_csv('data/top_anime_ranked_lists.csv')
anime_details_df = pd.read_csv("data/top_anime_details.csv")
score_vs_dropped_df = pd.read_csv("data/anime_score_vs_dropped.csv")
genres_friendships_df = pd.read_csv("data/genre_genre_friendship_counts.csv")
anime_runtime_df = pd.read_csv("data/anime_episodes_airtime_vs_score.csv")

top_m_genres_by_num = genre_top20_stats_df.sort_values(by='total_anime_count', ascending=False)[:MAX_NUM_OF_GENRES]

top_anime_details_df = pd.merge(top_anime_df, anime_details_df, on='anime_id', how='left')
top_anime_details_df["title_truncated"] = top_anime_details_df["title"].str.slice(0, 6) + "..."
score_vs_dropped_df["title_truncated"] = score_vs_dropped_df["title"].str.slice(0, 6) + "..."
correlation_details_df = pd.merge(score_vs_dropped_df, anime_details_df[['anime_id', 'main_pic']], on='anime_id', how='left')

# global variables
allowed_genres = top_m_genres_by_num["genre"].values.tolist()
list_of_genres = ["All genres"] + allowed_genres
list_of_metrics = ["completed_count", "favorites_count", "score"]

# for friendship graph only data of allowed genres
friendship_df = genres_friendships_df[
    genres_friendships_df["genre_1"].isin(allowed_genres) & genres_friendships_df["genre_2"].isin(allowed_genres)
].copy()

# runtime graph data 
# ----------------------------------------------------------------------------------
runtime_df = anime_runtime_df[
    ["title", "score", "num_episodes", "airing_weeks"]
].copy()

for col in ["score", "num_episodes", "airing_weeks"]:
    runtime_df[col] = pd.to_numeric(runtime_df[col], errors="coerce")

EP_BINS = list(range(0, 51, 5)) + [np.inf]   # 0,5,10,...,50,inf
EP_LABELS = [f"{i+1}–{i+5}" for i in range(0, 50, 5)] + ["50+"]

WK_BINS = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, np.inf]

WK_LABELS = [
    "1–4", "5–8", "9–12", "13–16", "17–20", "21–24",
    "25–28", "29–32", "33–36", "37–40", "41–44", "45–48",
    "49–52", "52+"
]

runtime_df["episodes_bin"] = pd.cut(
    runtime_df["num_episodes"],
    bins=EP_BINS,
    labels=EP_LABELS,
    include_lowest=True
)

runtime_df["weeks_bin"] = pd.cut(
    runtime_df["airing_weeks"],
    bins=WK_BINS,
    labels=WK_LABELS,
    include_lowest=True
)

runtime_agg_df = (
    runtime_df
    .dropna(subset=["episodes_bin", "weeks_bin"])
    .groupby(["weeks_bin", "episodes_bin"], observed=False)
    .agg(
        avg_score=("score", "mean"),
        count=("title", "count")
    )
    .reset_index()
)

pivot_avg = (
    runtime_agg_df
    .pivot(index="weeks_bin", columns="episodes_bin", values="avg_score")
    .reindex(index=WK_LABELS, columns=EP_LABELS)
)

pivot_cnt = (
    runtime_agg_df
    .pivot(index="weeks_bin", columns="episodes_bin", values="count")
    .reindex(index=WK_LABELS, columns=EP_LABELS)
)

# ----------------------------------------------------------------------------------

# metric name formatting to look better in graphs and tooltips
metric_name_display = {
    "completed_count": "Completed Count",
    "favorites_count": "Favorites Count",
    "score": "Score",
    "dropped_proportion": "Dropped Proportion"
}

#colorpalette from seaborn
#the color for all genres is the same as MyAnimeList
seaborn_colors = [MYANIMELIST_COLOR , '#3182bd', '#6baed6', "#4a6d80", '#c6dbef', '#e6550d', '#fd8d3c', '#fdae6b', '#fdd0a2', '#31a354', '#74c476', '#a1d99b', '#c7e9c0', '#756bb1', '#9e9ac8', '#bcbddc', "#4f4fe4", '#636363', '#969696', '#bdbdbd', "#754E4E"]
genre_color = dict(zip(list_of_genres, seaborn_colors))
genre_color["__ALL__"] = genre_color.pop("All genres")

# colors for heatmap
mal_gradient_colors = [
    "#f7f7f7",
    "#c6dbef",
    "#6baed6",
    "#3182bd",
    MYANIMELIST_COLOR,
    "#4a6d80",
]

def make_colorscale(colors):
    n = len(colors)
    return [[i / (n - 1), c] for i, c in enumerate(colors)]

colorscale_heatmap = make_colorscale(mal_gradient_colors)
