"""
E-Commerce — Brand Grouping

Query NL: "Group by brand and by color return the ratio between topwear 
           (apparel and accessories that are worn above the waist) and 
           bottomwear (worn at and below the waist)"

group_cols: ["color", LLM("productDisplayName, imageFile")]
agg_cols: [LLM("productDisplayName")]
semantic group: mixed (color is direct, brand inferred from display name and image)
semantic agg: yes (clothing category inferred from product name/image)
"""

import pandas as pd

def topwear_bottomwear_ratio(series):
    topwear_count = (series == "topwear").sum()
    bottomwear_count = (series == "bottomwear").sum()
    if bottomwear_count == 0:
        return float('inf') if topwear_count > 0 else 0
    return topwear_count / bottomwear_count

df = pd.read_csv("ecommerce_products.csv")
# assume columns: productID, brand, productDisplayName, productColor (LLM inferred), clothingCategory (LLM inferred: topwear/bottomwear)

# Group by brand and color, compute ratio
result = (
    df
    .groupby(["brand", "baseColour"])
    .agg({"subCategory": topwear_bottomwear_ratio})
    .reset_index()
    .rename(columns={"subCategory": "topwear_bottomwear_ratio"})
)

result.to_csv("ecommerce_2.csv", index=False)

#TODO: augmenting the brand to styles.csv 