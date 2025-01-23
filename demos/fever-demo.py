import argparse
import json
import os
import random
from pathlib import Path

from palimpzest.constants import Model
from palimpzest.core.data.datasources import ValidationDataSource
from palimpzest.core.elements.records import DataRecord
from palimpzest.core.lib.fields import BooleanField, ListField, StringField
from palimpzest.core.lib.schemas import Schema
from palimpzest.datamanager.datamanager import DataDirectory
from palimpzest.policy import MaxQuality, MinCost, MinTime
from palimpzest.query.processor.config import QueryProcessorConfig
from palimpzest.sets import Dataset
from palimpzest.utils.model_helpers import get_models
from ragatouille import RAGPretrainedModel


class FeverClaimsSchema(Schema):
    claim = StringField(desc="the claim being made")

class FeverIntermediateSchema(FeverClaimsSchema):
    relevant_wikipedia_articles = ListField(desc="Most relevant wikipedia articles to the `claim`",
                                            element_type=StringField)

class FeverOutputSchema(FeverIntermediateSchema):
    label = BooleanField("Output TRUE if the `claim` is supported by the evidence in `relevant_wikipedia_articles`; output FALSE otherwise.")


def get_label_fields_to_values(claims, ground_truth_file):
    with open(ground_truth_file) as f:
        ground_truth = [json.loads(line) for line in f]

    claim_to_label = {}

    for entry in ground_truth:
        if str(entry["id"]) in claims:
            evidence_sets = entry["evidence"]
            evidence_file_ids = list(
                {
                    evidence[2]
                    for evidence_set in evidence_sets
                    for evidence in evidence_set
                }
            )
            if entry["label"] == "SUPPORTS":
                claim_to_label[str(entry["id"])] = {"label": "TRUE"}
            else:
                claim_to_label[str(entry["id"])] = {"label": "FALSE"}
            claim_to_label[str(entry["id"])]["_evidence_file_ids"] = evidence_file_ids
            claim_to_label[str(entry["id"])]["relevant_wikipedia_articles"] = ["IGNORED_FIELD"]

    return claim_to_label           

class FeverValidationSource(ValidationDataSource):
    def __init__(self, dataset_id, claims_dir, split_idx: int=25, num_samples: int=5, shuffle: bool=False, seed: int=42):
        super().__init__(FeverClaimsSchema, dataset_id)
        self.claims_dir = claims_dir
        self.split_idx = split_idx
        self.claims = os.listdir(self.claims_dir)

        # shuffle records if shuffle = True
        if shuffle:
            random.Random(seed).shuffle(self.claims)

        self.val_claims = self.claims[:split_idx]
        self.claims = self.claims[split_idx:]

        self.num_samples = num_samples
        self.shuffle = shuffle
        self.seed = seed

        if split_idx != 25:
            raise Exception("Currently must split on split_idx=25 for correctness")

        if num_samples > 25:
            raise Exception("We have not labelled more than the first 25 listings!")

        # construct mapping from claim --> label (field, value) pairs
        self.label_fields_to_values = get_label_fields_to_values(self.val_claims, "testdata/paper_test.jsonl")

        # trim to number of samples
        self.val_claims = self.val_claims[:num_samples]

    def copy(self):
        return FeverValidationSource(self.dataset_id, self.claims_dir, self.split_idx, self.num_samples, self.shuffle, self.seed)

    def __len__(self):
        return len(self.claims)

    def get_val_length(self):
        return len(self.val_claims)

    def get_size(self):
        return sum(file.stat().st_size for file in Path(self.claims_dir).rglob('*'))

    def get_field_to_metric_fn(self):
        def bool_eval(label, expected_label):
            return str(label).upper() == str(expected_label).upper()
        
        def skip_eval(label, expected_label):
            return 1

        def list_eval(label, expected_label):
            # print("label: ", label)
            # print("expected_label: ", expected_label)
            if len(expected_label) == 0:
                return 1

            return len(set(label).intersection(set(expected_label))) * 1.0 / len(expected_label)

        fields_to_metric_fn = {
            "label": bool_eval,
            "relevant_wikipedia_articles": skip_eval,
            "_evidence_file_ids": list_eval
        }

        return fields_to_metric_fn

    def get_item(self, idx: int, val: bool=False, include_label: bool=False):
        # fetch listing
        claim = self.claims[idx] if not val else self.val_claims[idx]

        # create data record
        dr = DataRecord(self.schema, source_id=claim)

        claim_file = os.path.join(self.claims_dir, claim)
        with open(claim_file, "rb") as f:
            dr.claim = f.read().decode("utf-8")

        # if requested, also return the label information
        if include_label:
            # augment data record with label info
            labels_dict = self.label_fields_to_values[claim]

            for field, value in labels_dict.items():
                setattr(dr, field, value)

        return dr


if os.getenv("OPENAI_API_KEY") is None and os.getenv("TOGETHER_API_KEY") is None:
        print("WARNING: Both OPENAI_API_KEY and TOGETHER_API_KEY are unset")

# params

# parse arguments
parser = argparse.ArgumentParser(description="Run a simple demo")
parser.add_argument(
    "--verbose", default=False, action="store_true", help="Print verbose output"
)
# parser.add_argument("--datasetid", type=str, help="The dataset id")
# parser.add_argument("--workload", type=str, help="The workload to run. One of enron, real-estate, medical-schema-matching.")
parser.add_argument(
    "--processing_strategy",
    type=str,
    help='The processing strategy to use. One of no_sentinel, mab_sentinel',
    default='no_sentinel',
)
parser.add_argument(
    "--execution_strategy",
    type=str,
    help='The execution strategy to use. One of sequential, pipelined_single_thread, pipelined_parallel',
    default='sequential',
)
parser.add_argument(
    "--policy",
    type=str,
    help="One of 'mincost', 'mintime', 'maxquality'",
    default='maxquality',
)
parser.add_argument(
    "--num-samples",
    type=int,
    help="Number of validation samples",
    default=5,
)
parser.add_argument(
    "--rank",
    type=int,
    help="Rank for low-rank MC",
    default=4,
)
parser.add_argument(
    "--model",
    type=str,
    help="One of 'gpt-4o', 'gpt-4o-mini', 'llama', 'mixtral'",
    default='gpt-4o',
)

args = parser.parse_args()

num_claims = 100
dataset_id = f"fever-{num_claims}"
workload = "fever"
num_docs = 1000

index_path = f".ragatouille/colbert/indexes/fever-articles-{num_claims}-{num_docs}-index"
index = RAGPretrainedModel.from_index(index_path)

rank=4
num_samples=10
k = 10

engine = args.engine

if engine == "sentinel":
    k = -1

executor = "parallel" if engine == "sentinel" else "sequential"
model = args.model
policy_type = "maxquality"

verbose = True
allow_code_synth = False

policy = MaxQuality()
if policy_type == "mincost":
    policy = MinCost()
elif policy_type == "mintime":
    policy = MinTime()
elif policy_type == "maxquality":
    policy = MaxQuality()
else:
    print("Policy not supported for this demo")
    exit(1)

# select optimization strategy and available models based on engine
optimization_strategy, available_models = None, None
if engine == "sentinel":
    optimization_strategy = OptimizationStrategy.PARETO
    available_models = get_models(include_vision=True)
else:
    model_str_to_model = {
        "gpt-4o": Model.GPT_4o,
        "gpt-4o-mini": Model.GPT_4o_MINI,
        "mixtral": Model.MIXTRAL,
        "llama": Model.LLAMA3,
    }
    model_str_to_vision_model = {
        "gpt-4o": Model.GPT_4o_V,
        "gpt-4o-mini": Model.GPT_4o_MINI_V,
        "mixtral": Model.LLAMA3_V,
        "llama": Model.LLAMA3_V,
    }
    optimization_strategy = OptimizationStrategy.NONE
    available_models = [model_str_to_model[model]] + [model_str_to_vision_model[model]]


# datasetid="real-estate-eval-100" for paper evaluation
data_filepath = f"testdata/{dataset_id}"
user_dataset_id = f"{dataset_id}-user"

# create and register validation data source
datasource = FeverValidationSource(
    datasetId=f"{user_dataset_id}",
    claims_dir=data_filepath,
    num_samples=num_samples,
    shuffle=False,
    seed=42,
)

DataDirectory().register_user_source(
    src=datasource,
    dataset_id=f"{user_dataset_id}",
)

claims = Dataset(user_dataset_id, schema=FeverClaimsSchema)
claims_and_relevant_files = claims.retrieve(
    output_schema=FeverIntermediateSchema,
    index=index,
    search_attr="claim",
    output_attr="relevant_wikipedia_articles",
    k=k
)
output = claims_and_relevant_files.convert(output_schema=FeverOutputSchema)

assert args.processing_strategy in ["no_sentinel", "mab_sentinel"], "We only support no_sentinel and mab_sentinel for this demo"

# execute pz plan
config = QueryProcessorConfig(
    policy=policy,
    nocache=True,
    available_models=available_models,
    optimization_strategy=optimization_strategy,
    processing_strategy=args.processing_strategy,
    execution_strategy=args.execution_strategy,
    rank=rank,
    verbose=verbose,
    allow_code_synth=allow_code_synth
)
records, execution_stats = output.run(config)

# create filepaths for records and stats
records_path = (
    f"opt-profiling-data/{workload}-rank-{rank}-num-samples-{num_samples}-records.json"
    if engine == "sentinel"
    else f"opt-profiling-data/{workload}-baseline-{model}-records.json"
)
stats_path = (
    f"opt-profiling-data/{workload}-rank-{rank}-num-samples-{num_samples}-profiling.json"
    if engine == "sentinel"
    else f"opt-profiling-data/{workload}-baseline-{model}-profiling.json"
)

record_jsons = []
for record in records:
    record_dict = record.as_dict()
    ### field_to_keep = ["claim", "id", "label"]
    ### record_dict = {k: v for k, v in record_dict.items() if k in fields_to_keep}
    record_jsons.append(record_dict)

with open(records_path, 'w') as f:
    json.dump(record_jsons, f)

# save statistics
execution_stats_dict = execution_stats.to_json()
with open(stats_path, "w") as f:
    json.dump(execution_stats_dict, f)
