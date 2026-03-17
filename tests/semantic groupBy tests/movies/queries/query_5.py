"""
Query 7 — Sentiment by Director and Genre (Templatable, Mixed Group + Semantic Agg)

Query NL: "For movies directed by {director} in the {genre} genre, group reviews by
           the emotional tone of the review (Enthusiastic, Measured, Disappointed) and
           count the number of reviews per tone"
- group_cols: ["director" (literal, filtered), "genre" (literal, filtered),
               LLM("reviewText") → emotional tone]
- agg_cols:   ["reviewId" (count)]
- semantic group: mixed  (director and genre are filter/literal; tone is semantic)
- semantic agg:   no

Ground truth approximation: map scoreSentiment + originalScore to ternary label.
"""

import pandas as pd
import sys

DIRECTOR = sys.argv[1] if len(sys.argv) > 1 else "Steven Spielberg"
GENRE    = sys.argv[2] if len(sys.argv) > 2 else "Adventure"

def approx_tone(row):
    sentiment = row["scoreSentiment"]
    score_str = str(row["originalScore"])
    # Parse scores like "4/5", "3.5/4", "A", "B+" — use sentiment as fallback
    if sentiment == "NEGATIVE":
        return "Disappointed"
    # Try to parse numeric score to detect Enthusiastic vs Measured
    try:
        parts = score_str.split("/")
        if len(parts) == 2:
            ratio = float(parts[0]) / float(parts[1])
            return "Enthusiastic" if ratio >= 0.8 else "Measured"
    except Exception:
        pass
    return "Measured"  # default for POSITIVE without parseable score

movies  = pd.read_csv("../movies.csv")
reviews = pd.read_csv("../movie_reviews.csv")

filtered_movies = movies[
    movies["director"].str.contains(DIRECTOR, na=False, case=False) &
    movies["genre"].str.contains(GENRE, na=False, case=False)
][["id"]]

merged = reviews.merge(filtered_movies, on="id", how="inner")
merged["emotionalTone"] = merged.apply(approx_tone, axis=1)

result = (
    merged
    .groupby("emotionalTone")
    .agg(review_count=("reviewId", "count"))
    .reset_index()
)
result["director"] = DIRECTOR
result["genre"]    = GENRE

result.to_csv("query5_ground_truth.csv", index=False)
print(f"Generated ground truth for {DIRECTOR} in {GENRE}: {len(result)} tone groups")
