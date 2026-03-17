"""
Query 6 — Most Positive Review by Director (Semantic GroupBy + Numeric Agg)

Query NL: "Group by director and find the most positive review per director"
- group_cols: ["director" (literal, from movies table)]
- agg_cols:   [max(normalizedScore) from originalScore]
- semantic group: no (director is a literal column)
- semantic agg: yes (LLM("reviewText") used in PZ to score sentiment)

Ground truth:
  1. Join movie_reviews with movies on id to get director per review.
  2. Drop records where originalScore is missing or unparseable.
  3. Normalize originalScore ("3.5/4", "4/5", etc.) to [0, 1].
  4. For each director, select the review with the highest normalized score.

do it for each director and compute the distance between the score of the most positive
review using sem_groupBy (LLM(reviewText)) to actual output from python (ground truth).

doing directionally better. (don't worry about the exact numbers, just want to see if
it's improving or not). Show that these optimisations can get better performance and
then bake it into the query optimiser. (put it into the PZ and show the optimiser
can pick the best one)
"""

import pandas as pd


def parse_score(score_str):
    """
    Parse scores like "3.5/4", "4/5", "1/10" into a float in [0, 1].
    Returns None if the string is missing or unparseable.
    """
    if pd.isna(score_str) or str(score_str).strip() == "":
        return None
    parts = str(score_str).strip().split("/")
    if len(parts) == 2:
        try:
            numerator = float(parts[0])
            denominator = float(parts[1])
            if denominator == 0:
                return None
            return numerator / denominator
        except ValueError:
            return None
    return None


reviews = pd.read_csv("../movie_reviews.csv")
movies  = pd.read_csv("../movies.csv")[["id", "director"]]

# Join to get director for each review
merged = reviews.merge(movies, on="id", how="left")

# Drop records with missing originalScore
merged = merged.dropna(subset=["originalScore"])
merged = merged[merged["originalScore"].str.strip() != ""]

# Normalize originalScore to [0, 1]
merged["normalizedScore"] = merged["originalScore"].apply(parse_score)

# Drop records where score could not be parsed
merged = merged.dropna(subset=["normalizedScore"])

# Drop records with missing director
merged = merged.dropna(subset=["director"])

# For each director, pick the review with the highest normalized score
result = (
    merged
    .sort_values("normalizedScore", ascending=False)
    .groupby("director", as_index=False)
    .first()[["director", "normalizedScore", "reviewText", "originalScore"]]
)

result = result.sort_values("director").reset_index(drop=True)

result.to_csv("query6_ground_truth.csv", index=False)
print(f"Generated ground truth with {len(result)} directors")