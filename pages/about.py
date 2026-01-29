import dash
from dash import html, dcc
from data import MAX_NUM_OF_GENRES

dash.register_page(__name__, path='/About')

markdown = dcc.Markdown(f'''
# Visualization of MyAnimeList                  
Welcome to our web visualization of MyAnimeList. On this website we visualize the data from a rating service for anime called MyAnimeList. 
                      
The goal is to transform raw anime data into clear, interactive visuals that help uncover patterns, trends, and relationships across genres, popularity metrics, episode counts, and user interactions. Instead of focusing on individual titles, the project looks at anime as a whole ecosystem — how genres relate to each other, how popularity is distributed, and how different characteristics shape the anime space.

This dashboard is intended for anime fans, data enthusiasts, and anyone curious about how large-scale community-driven data can reveal structure and trends within pop culture.
                       
## About MyAnimeList
On MyAnimeList users can rate an anime with a score from 1 to 10, select if they have it on to watch list, started watching it, finished it or dropped it before the last episode. They can also add other users as friends. Each anime can be also tagged with genres. In the visualizations we only show top {MAX_NUM_OF_GENRES} most popular genres. User can also select an anime as their favourite.



## Navigation
On top of the page you have buttons About, which takes you to this page with instructions how the page works. 
Next to it is the button Visualization, which will take you to the page with all the visualizations of the data.

On Visualization page, the navigation bar on top has also two dropdown menus: genre and metric. 
- With **genre** dropdown menu, you can filter the data displayed on some plots to be only for animes that are tagged with that genre. You can select between showing data for animes from all genres (default option), or one of the 10 genres with most animes.
- With **metric** dropdown menu, you can select how the popularity of each anime in plots is calculated. You can select between completed_count (how many users have watched until the last episode), favorites_count (how many users added the anime to the list of their favourite anime), score (average score users rated it on a scale from 1 to 10)

## Visualization page
On the page there are 5 plots. On each corner of the plot there is an `i` icon that will show description of the plot on hover.

**Plots:**
* **Friendships between users by their favorite genre:** located on the top left of the page. It shows relations between the favorite genre of the user and favourite genre of their friends. Each bubble represents an anime genre. The size of the bubble is based on the number of pair of users that are friends and have the same favourite genre. The more there are such pair for the genre, the bigger the bubble is. On the other hand, the thickness of lines that connect two genre bubbles depend on number of friendship between users that have different favourite genre. For example if there are more users, whose favourite genre is A, that are firends with users, whose favourite genre is B, than users from genre A that are friends with users from genre C, then the line between genre A and B will be thicker than the line between genre A and C. On hovering the connection or the node, it will show the exact number. If you click on a node, other plots will show data for that genre, just like if you select the genre in the dropdown menu on the menu bar on top of the page. Below the graph there is also a slidebar to filter the data - it will show only genres and connections with friendship counts above the selected percentile.
* **Popularity of genres by `metric`:** located on the top right of the page. It is a bubble chart where each bubble represents a genre. The size of the bubble is based on the value of chosen metric. The bigger the value, the bigger the bubble. If user chooses for example *completed count* for metric, then the genres that have anime with more users that completed it are going to have a bigger bubble. If you click on a bubble, other plots will show data for that genre, just like if you select the genre in the dropdown menu on the menu bar on top of the page.
* **Top 20 anime:** Located on the right side of the page. A barchart showing the 20 most popular anime in the selected genre or among all animes if no genre is selected. Each bar represents an anime and the height shows the value of the chosen metric. If you hover on a bar you will see details about the anime.
* **Average anime score by episode count and airing duration:** Located on bottom left. It is a heatmap with values aggregated into cells. It shows how number of episodes of the anime and its airing duration affect the score. The aim is to show wether the anime that are longer or those that ae more frequently aired have better scores or not.
* **`metric` vs Dropout rate – `genre`:** This plot shows the relationship between score of the anime proportion of users that dropped the anime (will not watch until the last episode) for the selected genre. Each dot represents an anime. On hover it will show the details about the anime.
''')


layout = html.Div(
    markdown,
    style={
        "paddingLeft": "1cm",
        "paddingRight": "1cm",
        "paddingTop": "0.5cm",
        "paddingBottom": "0.5cm",
    }
)