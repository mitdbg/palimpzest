"""
Query 6 — Sentiment and Top Critic Bias by Genre (Multi-Col, Semantic Group + Agg)

Query NL: "Group reviews by inferred genre of the movie and whether the reviewer is a top critic,
           and compute the fraction of positive reviews"
- group_cols: [LLM("reviewText") for the genre, "isTopCritic"]
- agg_cols:   [LLM("reviewText") for POSITIVE/NEGATIVE, frac_positive]
- semantic group: yes  (genre inferred from review text)
- semantic agg:   yes  (sentiment inferred from reviewText)

Ground truth obtained by joining to movies table for genre.
"""

import pandas as pd

def frac_positive(series):
    return (series == "POSITIVE").sum() / len(series) if len(series) > 0 else 0.0

movies  = pd.read_csv("../movies.csv")[["id", "genre"]]
reviews = pd.read_csv("../movie_reviews.csv").head(500)

merged = reviews.merge(movies, on="id", how="left")
# Coarsen multi-genre entries to primary genre
merged["primaryGenre"] = merged["genre"].str.split(",").str[0].str.strip()

result = (
    merged
    .dropna(subset=["primaryGenre", "isTopCritic"])
    .groupby(["primaryGenre", "isTopCritic"])
    .agg(
        frac_positive=("scoreSentiment", frac_positive),
        review_count=("scoreSentiment", "count"),
    )
    .reset_index()
)

result.to_csv("query4_ground_truth.csv", index=False)
print(f"Generated ground truth with {len(result)} genre-topcritic groups")
