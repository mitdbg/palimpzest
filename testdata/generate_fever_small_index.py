import json
import os

from ragatouille import RAGPretrainedModel
    
def load_data(in_folder, doc_ids):
    texts = []
    ids = []
    
    for file in os.listdir(in_folder):
    # for file in ["wiki-001.jsonl"]:
        if file.endswith(".jsonl"):
            with open(f"{in_folder}/{file}", "r") as f:
                for line in f:
                    entry = json.loads(line)
                    if entry["id"] and entry["id"] in doc_ids:                    
                        texts.append(entry["text"])
                        ids.append(entry["id"])
    return texts, ids

    
def main():
    dataset_folder = "testdata/fever-10"
    query_ids = os.listdir(dataset_folder)
    query_ids = [int(query_id) for query_id in query_ids]
    print(query_ids)
    
    ground_truth_file = "testdata/paper_test.jsonl"
    with open(ground_truth_file, "r") as f:
        ground_truth = [json.loads(line) for line in f]
    
    doc_ids = []
    for entry in ground_truth:
        if (entry["id"]) in query_ids:
            for evidence_set in entry["evidence"]:
                for evidence in evidence_set:
                    if evidence[2] is not None:
                        doc_ids.append(evidence[2])
        
    print(doc_ids)
        
    # Load the data 
    in_folder = "testdata/fever-articles"
    texts, ids = load_data(in_folder, doc_ids)
    
    print(texts)

    RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

    RAG.index(
        collection=texts, 
        document_ids=ids,
        index_name="fever-articles-10-index"
        )

if __name__ == "__main__":
    main()
