import argparse
import json
import os

from ragatouille import RAGPretrainedModel

import palimpzest as pz

fever_claims_cols = [
    {"name": "claim", "type": str, "desc": "the claim being made"}
]

fever_output_cols = [
    {"name": "label", "type": bool, "desc": "Output TRUE if the `claim` is supported by the evidence in `relevant_wikipedia_articles`; output FALSE otherwise."}
]

class FeverDataReader(pz.DataReader):
    def __init__(self, claims_file_path, num_claims_to_process):
        super().__init__(fever_claims_cols)

        # `claims_file_path` is the path to the file containing the claims which is expected to be a jsonl file.
        # Each line in the file is a JSON object with an "id" and a "claim" field.

        self.claims_file_path = claims_file_path
        self.num_claims_to_process = num_claims_to_process

        with open(claims_file_path) as f:
            entries = [json.loads(line) for line in f]
            entries = entries[: min(num_claims_to_process, len(entries))]
            self.claims = [entry["claim"] for entry in entries]
            self.ids = [entry["id"] for entry in entries]

    def __len__(self):
        return len(self.claims)

    def __getitem__(self, idx: int):
        # get claim
        claim = self.claims[idx]

        # construct and return dictionary with field(s)
        return {"claim": claim}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run FEVER demo")
    parser.add_argument("--claims-file-path", type=str, help="Path to the claims file")
    parser.add_argument(
        "--num-claims-to-process", type=int, help="Number of claims from the dataset to process", default=5
    )
    parser.add_argument("--index-path", type=str, help="Path to the index")
    parser.add_argument(
        "--k",
        type=int,
        help="Number of relevant documents to retrieve from the index (k for the k-nearest-neighbor lookup from the index)",
        default=5,
    )
    return parser.parse_args()


def build_fever_query(index, dataset, k):
    claims = pz.Dataset(dataset)
    claims = claims.sem_add_columns(fever_claims_cols, desc="Extract the claim")

    def search_func(index, query, k):
        results = index.search(query, k=k)
        return [result["content"] for result in results]

    claims_and_relevant_files = claims.retrieve(
        index=index,
        search_func=search_func,
        search_attr="claim",
        output_attr="relevant_wikipedia_articles",
        output_attr_desc="Most relevant wikipedia articles to the `claim`",
        k=k,
    )
    output = claims_and_relevant_files.sem_add_columns(fever_output_cols)
    return output


def main():
    if os.getenv("OPENAI_API_KEY") is None and os.getenv("TOGETHER_API_KEY") is None:
        print("WARNING: Both OPENAI_API_KEY and TOGETHER_API_KEY are unset")

    args = parse_arguments()

    # Create a data reader for the FEVER dataset
    dataset = FeverDataReader(
        claims_file_path=args.claims_file_path,
        num_claims_to_process=args.num_claims_to_process,
    )

    # Load the index
    index = RAGPretrainedModel.from_index(args.index_path)

    # Build and run the FEVER query
    query = build_fever_query(index, dataset, k=args.k)
    data_record_collection = query.run(pz.QueryProcessorConfig())
    print(data_record_collection.to_df())

if __name__ == "__main__":
    main()
