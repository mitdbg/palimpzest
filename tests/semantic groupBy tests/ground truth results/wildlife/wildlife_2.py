"""
Wildlife — Lat/Long Extraction 

Query NL: "Group by country (from the longitude and latitude). 
           Compute the count of {animal} for every country."

group_cols: [LLM("latitude", "longitude")]
agg_cols: [LLM("imageFile")]
semantic group: yes (country inferred from coordinates)
semantic agg: yes (animal type inferred from image)
"""

import pandas as pd

df = pd.read_csv("wildlife_location.csv")
# assume columns: animalID, latitude, longitude, imageFile, country (LLM inferred), animalType (LLM inferred from image)

ANIMAL_TYPE = "lion"

# Filter by animal type
filtered_df = df[df["animalType"] == ANIMAL_TYPE]

# Group by country and animal type, count animals
result = (
    filtered_df
    .groupby(["country", "animalType"])
    .agg({"animalID": "count"})
    .reset_index()
    .rename(columns={"animalID": "animal_count"})
)

result.to_csv("wildlife_2.csv", index=False)

#TODO: Augment country to the dataset