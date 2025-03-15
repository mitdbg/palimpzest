import json
import os
import time

import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import numpy as np
from openai import OpenAI
from tqdm import tqdm

# NOTE: this script is meant to be run from the root of the repository
if __name__ == "__main__":
    # initialize openai client
    openai_client = OpenAI()

    # load texts
    texts, text_ids = [], []
    with open("testdata/MMQA_texts.jsonl") as f:
        for line in f:
            dict_line = json.loads(line)
            title = dict_line["title"]
            text = dict_line["text"]
            texts.append(f"{title}: {text}")
            text_ids.append(dict_line["id"])

    # create directory for embeddings
    os.makedirs("testdata/mmqa-text-embeddings/", exist_ok=True)

    # generate embeddings in batches of 1000 at a time
    indices = np.linspace(0, len(texts), len(texts)//1000, dtype=int)
    total_embeds = len(indices)
    print(f"Generating {total_embeds} embeddings...")
    gen_indices = []
    for iter_idx, start_idx in tqdm(enumerate(indices), total=total_embeds):
        # check if embedding needs to be computed
        end_idx = indices[iter_idx + 1] if iter_idx + 1 < len(indices) else None
        filename = f"testdata/mmqa-text-embeddings/{start_idx}_{end_idx}.npy"
        if end_idx is not None and not os.path.exists(filename):
            # generate embeddings
            batch = texts[start_idx:end_idx]
            resp = openai_client.embeddings.create(input=batch, model="text-embedding-3-small")
            embeddings = [item.embedding for item in resp.data]

            # save embeddings to disk
            with open(filename, "wb") as f:
                np.save(f, np.array(embeddings))

            gen_indices.append((start_idx, end_idx))
            time.sleep(1)
    print("Done generating embeddings.")

    # initialize chroma client
    chroma_client = chromadb.PersistentClient(".chroma-mmqa")

    # initialize embedding function
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"],
        model_name="text-embedding-3-small"
    )

    # create a collection
    collection = chroma_client.get_or_create_collection(
        name="mmqa-texts",
        embedding_function=openai_ef,
        metadata={"hnsw:space": "cosine"},
    )

    # insert documents in batches
    total_inserts = len(gen_indices)
    print(f"Inserting {total_inserts} batches into the collection...")
    for start_idx, end_idx in tqdm(gen_indices, total=total_inserts):
        embeddings = np.load(f"testdata/mmqa-text-embeddings/{start_idx}_{end_idx}.npy")
        collection.add(
            documents=texts[start_idx:end_idx],
            embeddings=embeddings.tolist(),
            ids=text_ids[start_idx:end_idx],
        )
