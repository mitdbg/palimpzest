import json
import os

import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

CORRUPTED_IMAGE_IDS = [
    "17ae0616ac745e70781203267f3a382d",
    "bf201cbbd058ef51aef89b1be4158c2a",
]

# NOTE: this script is meant to be run from the root of the repository
if __name__ == "__main__":
    # load CLIP model
    model = SentenceTransformer("clip-ViT-B-32")

    # load image metadata
    image_filepaths, image_ids = [], []
    with open("testdata/MMQA_images.jsonl") as f:
        possible_endings = {'.JPG', '.png', '.jpeg', '.jpg', '.tif', '.JPEG', '.tiff', '.PNG', '.Jpg', '.gif'}
        for line in f:
            dict_line = json.loads(line)
            image_id = dict_line["id"]

            # skip corrupted images:
            if image_id in CORRUPTED_IMAGE_IDS:
                continue

            # add image to image_ids
            image_ids.append(image_id)

            # find the correct image file
            for ending in possible_endings:
                if os.path.exists(f"testdata/mmqa-images/{image_id}{ending}"):
                    image_id += ending
                    break

            image_filepaths.append(f"testdata/mmqa-images/{image_id}")

    # create directory for embeddings
    os.makedirs("testdata/mmqa-image-embeddings/", exist_ok=True)

    # generate embeddings in batches of 128 at a time
    indices = np.linspace(0, len(image_filepaths), len(image_filepaths)//128, dtype=int)
    total_embeds = len(indices)
    print(f"Generating {total_embeds} batches of embeddings...")
    gen_indices = []
    for iter_idx, start_idx in tqdm(enumerate(indices), total=total_embeds):
        # check if embedding needs to be computed
        end_idx = indices[iter_idx + 1] if iter_idx + 1 < len(indices) else None
        filename = f"testdata/mmqa-image-embeddings/{start_idx}_{end_idx}.npy"
        if end_idx is not None and not os.path.exists(filename):
            # generate embeddings
            batch_fps = image_filepaths[start_idx:end_idx]
            batch_images = [Image.open(fp) for fp in batch_fps]
            embeddings = model.encode(batch_images)

            # save embeddings to disk
            with open(filename, "wb") as f:
                np.save(f, embeddings)

            gen_indices.append((start_idx, end_idx))
    print("Done generating embeddings.")

    # initialize chroma client
    chroma_client = chromadb.PersistentClient(".chroma-mmqa")

    # initialize embedding function
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="clip-ViT-B-32"
    )

    # create a collection
    collection = chroma_client.get_or_create_collection(
        name="mmqa-images",
        embedding_function=sentence_transformer_ef,
        metadata={"hnsw:space": "cosine"},
    )

    # insert documents in batches
    total_inserts = len(gen_indices)
    print(f"Inserting {total_inserts} batches into the collection...")
    for start_idx, end_idx in tqdm(gen_indices, total=total_inserts):
        embeddings = np.load(f"testdata/mmqa-image-embeddings/{start_idx}_{end_idx}.npy")
        collection.add(
            documents=[os.path.basename(fp) for fp in image_filepaths[start_idx:end_idx]],
            embeddings=embeddings.tolist(),
            ids=image_ids[start_idx:end_idx],
        )
