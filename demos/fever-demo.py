import argparse
import json
import os

import pandas as pd
from ragatouille import RAGPretrainedModel

from palimpzest.core.data.datasources import UserSource
from palimpzest.core.elements.records import DataRecord
from palimpzest.core.lib.fields import BooleanField, StringField
from palimpzest.core.lib.schemas import Schema
from palimpzest.datamanager.datamanager import DataDirectory
from palimpzest.query.processor.config import QueryProcessorConfig
from palimpzest.sets import Dataset


class FeverClaimsSchema(Schema):
    claim = StringField(desc="the claim being made")

class FeverOutputSchema(FeverClaimsSchema):
    label = BooleanField(
        "Output TRUE if the `claim` is supported by the evidence in `relevant_wikipedia_articles`; output FALSE otherwise."
    )


class FeverUserSource(UserSource):
    def __init__(self, dataset_id, claims_file_path, num_claims_to_process):
        super().__init__(FeverClaimsSchema, dataset_id)

        # `claims_file_path` is the path to the file containing the claims which is expected to be a jsonl file.
        # Each line in the file is a JSON object with an "id" and a "claim" field.

        self.claims_file_path = claims_file_path
        self.num_claims_to_process = num_claims_to_process

        with open(claims_file_path) as f:
            entries = [json.loads(line) for line in f]
            entries = entries[: min(num_claims_to_process, len(entries))]
            self.claims = [entry["claim"] for entry in entries]
            self.ids = [entry["id"] for entry in entries]

    def copy(self):
        return FeverUserSource(self.dataset_id, self.claims_file_path, self.num_claims_to_process)

    def __len__(self):
        return len(self.claims)

    def get_size(self):
        return sum([len(claim) for claim in self.claims])

    def get_item(self, idx: int):
        claim = self.claims[idx]
        dr = DataRecord(self.schema, source_id=self.ids[idx])
        dr.claim = claim
        return dr


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


def build_fever_query(index, dataset_id, k):
    claims = Dataset(dataset_id, schema=FeverClaimsSchema)
    
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
    output = claims_and_relevant_files.convert(output_schema=FeverOutputSchema)
    return output


def main():
    if os.getenv("OPENAI_API_KEY") is None and os.getenv("TOGETHER_API_KEY") is None:
        print("WARNING: Both OPENAI_API_KEY and TOGETHER_API_KEY are unset")

    args = parse_arguments()

    # Create a user datasource for the FEVER dataset
    dataset_id = f"fever-dataset-{args.num_claims_to_process}"
    datasource = FeverUserSource(
        dataset_id=dataset_id,
        claims_file_path=args.claims_file_path,
        num_claims_to_process=args.num_claims_to_process,
    )
    DataDirectory().register_user_source(
        src=datasource,
        dataset_id=dataset_id,
    )

    # Load the index
    index = RAGPretrainedModel.from_index(args.index_path)

    # Build and run the FEVER query
    query = build_fever_query(index, dataset_id, k=args.k)
    results, execution_stats = query.run(QueryProcessorConfig())

    output_df = pd.DataFrame([r.to_dict() for r in results])[["claim", "relevant_wikipedia_articles", "label"]]
    print(output_df)


if __name__ == "__main__":
    main()
