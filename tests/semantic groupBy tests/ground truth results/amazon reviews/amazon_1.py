"""
Amazon Sales — Review Analysis

Query NL: "Group by review type and return average cost of the products"

group_cols: [LLM("reviewText")]
agg_cols: ["price"]
semantic group: yes (review type/sentiment inferred from review text)
semantic agg: no (average is a standard aggregate)
"""

import pandas as pd

df = pd.read_csv("amazon.csv")
# assume columns: productID, reviewText, price, reviewType (LLM inferred: positive/negative/neutral)

# Group by review type and compute average price
result = (
    df
    .groupby("reviewType")
    .agg({"price": "mean"})
    .reset_index()
    .rename(columns={"price": "avg_price"})
)

result.to_csv("amazon-review-type-avg-price.csv", index=False)