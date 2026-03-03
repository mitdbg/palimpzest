"""
Query 2 — Critic Volume by Inferred Era (Single Col, Semantic Group)

Query NL: "Group reviews by the era of the movie they reviewed (pre-2000, 2000s, 2010s, 2020s)
           and count the number of reviews per era"
- group_cols: [LLM("reviewDate")]
- agg_cols: ["reviewId" (count)]
- semantic group: yes
- semantic agg: no

Ground truth uses date parsing and rule-based era bucketing.
"""

import pandas as pd

reviews = pd.read_csv("../movie_reviews.csv").head(500)
movies  = pd.read_csv("../movies.csv")[["id", "releaseDateTheaters"]]

# Join to get the movie's release year
merged = reviews.merge(movies, on="id", how="left")
merged["releaseYear"] = pd.to_datetime(
    merged["releaseDateTheaters"], errors="coerce"
).dt.year

def era_bucket(year):
    if pd.isna(year):   return "Unknown"
    if year < 2000:     return "pre-2000"
    if year < 2010:     return "2000s"
    if year < 2020:     return "2010s"
    return "2020s"

merged["era"] = merged["releaseYear"].apply(era_bucket)

result = (
    merged
    .groupby("era")
    .agg(review_count=("reviewId", "count"))
    .reset_index()
)

result.to_csv("query2_ground_truth.csv", index=False)
print(f"Generated ground truth with {len(result)} era groups")
