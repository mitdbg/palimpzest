import json
import os
import random

from ragatouille import RAGPretrainedModel
    
def load_data(in_folder, doc_ids, num_docs):
    texts = []
    ids = []
    
    id_to_text = {}
    for file in os.listdir(in_folder):
        if file.endswith(".jsonl"):
            with open(f"{in_folder}/{file}", "r") as f:
                for line in f:
                    entry = json.loads(line)
                    id_to_text[entry["id"]] = entry["text"]
                    
    random_ids = random.sample(sorted(id_to_text), num_docs)
    
    for id in doc_ids:
        if id not in id_to_text:
            print(f"Missing {id}")
            continue
        ids.append(id)
        texts.append(id_to_text[id])
        
    for id in random_ids:
        if id not in set(doc_ids):
            ids.append(id)
            texts.append(id_to_text[id])
        
    # for file in os.listdir(in_folder):
    # # for file in ["wiki-001.jsonl"]:
    #     if file.endswith(".jsonl"):
    #         with open(f"{in_folder}/{file}", "r") as f:
    #             for line in f:
    #                 entry = json.loads(line)
    #                 if entry["id"] and entry["id"] in doc_ids:                    
    #                     texts.append(entry["text"])
    #                     ids.append(entry["id"])
    
    return texts, ids

    
def gen_index(sample_size, num_docs):
    dataset_folder = f"testdata/fever-{sample_size}"
    query_ids = os.listdir(dataset_folder)
    query_ids = [int(query_id) for query_id in query_ids]
    # print(query_ids)
    
    ground_truth_file = "testdata/paper_test.jsonl"
    with open(ground_truth_file, "r") as f:
        ground_truth = [json.loads(line) for line in f]
    
    doc_ids = set()
    for entry in ground_truth:
        if (entry["id"]) in set(query_ids):
            for evidence_set in entry["evidence"]:
                for evidence in evidence_set:
                    if evidence[2] is not None:
                        doc_ids.add(evidence[2])
        
    # print(doc_ids)
        
    # Load the data 
    in_folder = "testdata/fever-articles"
    texts, ids = load_data(in_folder, doc_ids, num_docs)
    
    print(f"Indexing {len(texts)} files")

    RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

    RAG.index(
        collection=texts, 
        document_ids=ids,
        index_name=f"fever-articles-{sample_size}-{num_docs}-index",
        split_documents=False)

if __name__ == "__main__":
    gen_index(sample_size=100, num_docs=1000)
