"""
Query 1 — Sentiment by Publication (Single Col, Semantic Agg)

Query NL: "Group by publicationName and compute the fraction of positive reviews"
- group_cols: ["publicationName"]
- agg_cols: [LLM("reviewText") for POSITIVE/NEGATIVE]
- semantic group: no
- semantic agg: yes

Ground truth from scoreSentiment column.
"""

import pandas as pd

def frac_positive(series):
    num_pos = (series == "POSITIVE").sum()
    return num_pos / len(series) if len(series) > 0 else 0.0

reviews = pd.read_csv("../movie_reviews.csv").head(500)

result = (
    reviews
    .groupby("publicatioName")
    .agg(frac_positive_sentiment=("scoreSentiment", frac_positive))
    .reset_index()
    .rename(columns={"frac_positive_sentiment": "frac_positive"})
)

result.to_csv("query1_ground_truth.csv", index=False)
print(f"Generated ground truth with {len(result)} publication groups")
