import json
import os

from ragatouille import RAGPretrainedModel
    
def load_data(in_folder):
    texts = []
    ids = []
    
    for file in os.listdir(in_folder):
    # for file in ["wiki-001.jsonl"]:
        if file.endswith(".jsonl"):
            with open(f"{in_folder}/{file}", "r") as f:
                for line in f:
                    entry = json.loads(line)
                    if entry["id"]:                    
                        texts.append(entry["text"])
                        ids.append(entry["id"])
    return texts, ids

    
def main():
    # Load the data 
    in_folder = "testdata/fever-articles"
    texts, ids = load_data(in_folder)
    
    print(texts[:2], ids[:2])

    RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

    RAG.index(
        collection=texts, 
        document_ids=ids,
        index_name="fever-articles-index"
        )

if __name__ == "__main__":
    main()
