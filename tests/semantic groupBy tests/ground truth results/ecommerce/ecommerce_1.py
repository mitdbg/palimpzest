"""
E-Commerce — Color Analysis 

Query NL: "Group by color of images and return the count"

group_cols: [LLM("imageFile")]
agg_cols: ["productID"]
semantic group: yes (color inferred from product image)
semantic agg: no 
"""

import pandas as pd

df = pd.read_csv("ecommerce_products.csv")
# assume columns: productID, imageFile, productColor (LLM inferred from image)

# Group by color and count products
result = (
    df
    .groupby("baseColour")
    .agg({"productID": "count"})
    .reset_index()
    .rename(columns={"productID": "product_count"})
)

result.to_csv("ecommerce_1.csv", index=False)

#TODO: join images.csv and styles.csv by productID to get imageFile and productColor