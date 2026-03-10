"""
Movies - Sentiment Analysis 

Query NL: "Group by criticName and compute the fraction of reviews with positive sentiment"
- group_cols: ["criticName"]
- agg_cols: [LLM("reviewText")]
- semantic group: no
- semantic agg: yes
"""

import pandas as pd

def frac_positive(series):
  num_pos = (series == "POSITIVE").sum()
  total = len(series)
  return num_pos / total

df = pd.read_csv("movie_reviews.csv")
# assume columns: criticName, reviewText, scoreSentiment

result = (
    df
    .groupby("criticName")
    .agg({"scoreSentiment": frac_positive})
    .reset_index()
)

result.to_csv("movies_1.csv", index=False)