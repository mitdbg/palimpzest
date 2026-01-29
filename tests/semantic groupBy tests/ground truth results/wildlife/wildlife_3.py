"""
Wildlife — Average Age

Query NL: "Group by small animals (from image) and return their average age."
Note: Small = an animal that weighs less than 50kg and has dimensions less than 1m

group_cols: [LLM("imageFile")]
agg_cols: ["age"]
semantic group: yes (size category inferred from image, weight and dimensions)
semantic agg: no 
"""

import pandas as pd

df = pd.read_csv("wildlife_detailed.csv")
# assume columns: animalID, imageFile, age, weight_kg, max_dimension_m, isSmall (LLM inferred: weight < 50kg AND dimension < 1m)

# Filter by small animals (LLM output already materialized)
small_animals_df = df[df["isSmall"] == True]

# Calculate average age
result = pd.DataFrame({
    "size_category": ["small"],
    "avg_age": [small_animals_df["age"].mean()]
})

result.to_csv("wildlife_3.csv", index=False)

# TODO: Augment size_category to the dataset