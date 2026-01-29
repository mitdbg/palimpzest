"""
Amazon Sales — Product Sentiment 

Query NL: "Group by user product review title"
Categories:
- Good overall
- Neutral 
- Bad overall

group_cols: [LLM("reviewTitle")]
agg_cols: ["productID"]
semantic group: yes (sentiment category inferred from review title)
semantic agg: no 
"""

import pandas as pd

df = pd.read_csv("amazon_sales.csv")
# assume columns: productID, reviewTitle, sentimentCategory (LLM inferred: good_overall/good_with_negatives/bad_with_positives/bad_overall)

# Group by sentiment category and count products
result = (
    df
    .groupby("sentimentCategory")
    .agg({"productID": "count"})
    .reset_index()
    .rename(columns={"productID": "product_count"})
)

result.to_csv("amazon-sentiment-category-count.csv", index=False)
