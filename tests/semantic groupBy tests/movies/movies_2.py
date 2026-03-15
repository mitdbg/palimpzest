"""
Movies — Templated Query 

Query NL: "Group by director and genre, and count movies with directed by {director} in {genre}."
Categories:
- Adventure
- Action 
- Comedy
- Mystery/Crime 
- Fantasy 
- Horror
- Romance 
- Sci-fi 

group_cols: [Director, LLM("Genre", "reviewText")]
agg_cols: []
semantic group: mixed (director name is literal, genre inferred from movie metadata)
semantic agg: no 
"""

import pandas as pd

# Parameters for the templated query
DIRECTOR = "Christopher Nolan"
GENRE = "Science Fiction"

df = pd.read_csv("movies_reviews.csv")
# assume columns: Director, Genre, reviewText, scoreSentiment, movieTitle

# Filter by director and genre
filtered_df = df[
    (df["Director"] == DIRECTOR) & 
    (df["Genre"] == GENRE) 
]

# Group by Director and Genre, count the number of movies
result = (
    filtered_df
    .groupby(["Director", "Genre"])
    .agg({"movieTitle": "count"})
    .reset_index()
    .rename(columns={"movieTitle": "movie_count"})
)

result.to_csv("movies_2.csv", index=False)

# TODO: Augment genre to the dataset 
# TODO: join the datasets 
