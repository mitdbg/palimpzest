"""
Query 3 — Fraction Positive per Audience Type (Templatable, Semantic Group)

Query NL: "For movies directed by {director}, group reviews by the audience type targeted
           by the movie's MPAA rating (Children, Teen, Adult, Unrated) and compute the
           fraction of positive reviews per audience type"
- group_cols: [LLM("rating") → audience type]
- agg_cols: [LLM("reviewText") → POSITIVE/NEGATIVE, frac_positive]
- semantic group: yes
- semantic agg: yes

Ground truth uses MPAA rating mapping and scoreSentiment column.
"""

import pandas as pd
import sys

DIRECTOR = sys.argv[1] if len(sys.argv) > 1 else "Christopher Nolan"

RATING_TO_AUDIENCE = {
    "G": "Children", "PG": "Children",
    "PG-13": "Teen",
    "R": "Adult", "NC-17": "Adult",
    "NR": "Unrated", "": "Unrated",
}

def frac_positive(series):
    return (series == "POSITIVE").sum() / len(series) if len(series) > 0 else 0.0

movies  = pd.read_csv("../movies.csv")
reviews = pd.read_csv("../movie_reviews.csv")

# Filter for director's movies
director_movies = movies[movies["director"].str.contains(DIRECTOR, na=False, case=False)][["id", "rating"]]
director_movies["audienceType"] = director_movies["rating"].map(
    lambda r: RATING_TO_AUDIENCE.get(str(r).strip(), "Unrated")
)

# merged = reviews.merge(director_movies, on="id", how="inner")

print("director_movies shape:", director_movies.shape)
print(director_movies.head())

merged = director_movies.merge(reviews, on="id", how="left")
print("merged shape:", merged.shape)
print(merged.head())

result = (
    merged
    .groupby("audienceType")
    .agg(
        frac_positive=("scoreSentiment", frac_positive),
        review_count=("scoreSentiment", "count"),
    )
    .reset_index()
)
result["director"] = DIRECTOR

result.to_csv("query3_ground_truth.csv", index=False)
print(f"Generated ground truth for {DIRECTOR}: {len(result)} audience type groups")
