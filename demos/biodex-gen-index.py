import os
import time

import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import numpy as np
from tqdm import tqdm

# NOTE: this script is meant to be run from the root of the repository
if __name__ == "__main__":
    # initialize chroma client
    chroma_client = chromadb.PersistentClient(".chroma")

    # initialize embedding function
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"],
        model_name="text-embedding-3-small"
    )

    # create a collection
    collection = chroma_client.create_collection(name="biodex-reaction-terms", embedding_function=openai_ef)

    # load reaction terms
    reaction_terms = []
    with open("testdata/reaction_terms.txt") as f:
        for line in f:
            reaction_terms.append(line.strip())

    # insert documents slowly (~10 at a time)
    indices = np.linspace(0, len(reaction_terms), len(reaction_terms)//10, dtype=int)
    total_inserts = len(indices)
    for iter_idx, start_idx in tqdm(enumerate(indices)):
        if iter_idx + 1 < len(indices):
            end_idx = indices[iter_idx + 1]
            collection.upsert(
                documents=reaction_terms[start_idx:end_idx],
                ids=[f"id{idx}" for idx in range(start_idx, end_idx)]
            )
        time.sleep(1)
