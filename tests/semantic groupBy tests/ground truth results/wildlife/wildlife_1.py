"""
Wildlife — Audio-to-Logic 

Query NL: "Group by animals that are carnivorous (from audio) and return the count for all such animals."

group_cols: [LLM("audioFile")]
agg_cols: ["animalID"]
semantic group: yes (diet type inferred from audio)
semantic agg: no 
"""

import pandas as pd

df = pd.read_csv("wildlife_audio.csv")
# assume columns: animalID, animalName, audioFile, dietType (LLM inferred from audio)

# Filter by carnivorous animals (LLM output already materialized)
carnivorous_df = df[df["dietType"] == "carnivorous"]

# Count the number of carnivorous animals
result = pd.DataFrame({
    "dietType": ["carnivorous"],
    "animal_count": [len(carnivorous_df)]
})

result.to_csv("wildlife_1.csv", index=False)

#TODO: Augment dietType to the dataset