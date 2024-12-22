import argparse
import json
import os
import random
from pathlib import Path

from ragatouille import RAGPretrainedModel

import palimpzest as pz
from palimpzest.constants import Model
from palimpzest.utils.model_helpers import get_models


class FeverClaimsSchema(pz.Schema):
    claim = pz.StringField(desc="the claim being made")

class FeverIntermediateSchema(FeverClaimsSchema):
    relevant_wikipedia_articles = pz.ListField(desc="Most relevant wikipedia articles to the `claim`",
                                               element_type=pz.StringField, required=True)

# class FeverSummarySchema(FeverIntermediateSchema):
#     summary = pz.StringField(desc="summary of the relevant wikipedia articles `file1`, `file2`, and `file3` to validate `claim`")

class FeverOutputSchema(FeverIntermediateSchema):
    label = pz.BooleanField("Output TRUE if the `claim` is supported by the evidence in `relevant_wikipedia_articles`; output FALSE otherwise.")

# class FeverOutputSchema(FeverSummarySchema):
#     label = pz.BooleanField("Output TRUE if the `claim` is supported by the evidence in `file1`, `file2`, and `file3`; output FALSE otherwise.")

# def get_relevant_content(index, k, record):
#     # input: record of type FeverInputSchema
#     # output: record of type FeverIntermediateSchema
#     relevant_files = index.search(record.claim , k=k)
#     most_relevant_files = [relevant_file["content"] for relevant_file in relevant_files]
   
#     # create output DataRecord
#     out_record = pz.DataRecord.from_parent(FeverIntermediateSchema, parent_record=record)
    
#     out_record.relevant_wikipedia_articles = most_relevant_files
#     # if len(most_relevant_files) < 3:
#     #     most_relevant_files += [""] * (3 - len(most_relevant_files))
        
#     # out_record.file1 = most_relevant_files[0]
#     # out_record.file2 = most_relevant_files[1]
#     # out_record.file3 = most_relevant_files[2]
#     return out_record

# def set_input_schema(input: DataRecord):
#     # , project_cols=[]
#     output = DataRecord.from_parent(FeverClaimsSchema, parent_record=input)
#     output.claim = input.contents
#     return output

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

    # print(claims)
    # print(claim_to_label)

    # assert(len(claim_to_label) == len(claims))

    # print(claim_to_label)

    return claim_to_label           

class FeverValidationSource(pz.ValidationDataSource):
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
        dr = pz.DataRecord(self.schema, source_id=claim)

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


if "DSP_CACHEBOOL" not in os.environ or os.environ["DSP_CACHEBOOL"].lower() != "false":
        raise Exception("TURN OFF DSPy CACHE BY SETTING `export DSP_CACHEBOOL=False")

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
    "--engine",
    type=str,
    help='The engine to use. One of sentinel, nosentinel',
    default='sentinel',
)
parser.add_argument(
    "--executor",
    type=str,
    help='The plan executor to use. One of sequential, pipelined, parallel',
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

policy = pz.MaxQuality()
if policy_type == "mincost":
    policy = pz.MinCost()
elif policy_type == "mintime":
    policy = pz.MinTime()
elif policy_type == "maxquality":
    policy = pz.MaxQuality()
else:
    print("Policy not supported for this demo")
    exit(1)

if engine == "sentinel":
    if executor == "sequential":
        execution_engine = pz.SequentialSingleThreadSentinelExecution
    elif executor == "parallel":
        execution_engine = pz.SequentialParallelSentinelExecution
    else:
        print("Unknown executor")
        exit(1)
elif engine == "nosentinel":
    if executor == "sequential":
        execution_engine = pz.SequentialSingleThreadNoSentinelExecution
    elif executor == "pipelined":
        execution_engine = pz.PipelinedSingleThreadNoSentinelExecution
    elif executor == "parallel":
        execution_engine = pz.PipelinedParallelNoSentinelExecution
    else:
        print("Unknown executor")
        exit(1)
else:
    print("Unknown engine")
    exit(1)

# select optimization strategy and available models based on engine
optimization_strategy, available_models = None, None
if engine == "sentinel":
    optimization_strategy = pz.OptimizationStrategy.PARETO
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
    optimization_strategy = pz.OptimizationStrategy.NONE
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

pz.DataDirectory().register_user_source(
    src=datasource,
    dataset_id=f"{user_dataset_id}",
)

claims = pz.Dataset(user_dataset_id, schema=FeverClaimsSchema)

# claims = pz.Dataset(dataset_id, schema=pz.TextFile,
# desc="Contains the claims that need to be verified.")
# claims = claims.convert(output_schema=FeverClaimsSchema, udf=set_input_schema)
# claims_and_relevant_files = claims.convert(output_schema=FeverIntermediateSchema,
#                                            udf=partial(get_relevant_content, index, k))
# claims_and_relevant_files = claims.retrieve(
#     output_schema=FeverIntermediateSchema,
#     index=index,
#     search_attr="claim",
#     output_attr="relevant_wikipedia_articles",
#     k=10,
# )

claims_and_relevant_files = claims.retrieve(
    output_schema=FeverIntermediateSchema,
    index=index,
    search_attr="claim",
    output_attr="relevant_wikipedia_articles",
    k=k
)
output = claims_and_relevant_files.convert(output_schema=FeverOutputSchema)

# execute pz plan

records, execution_stats = pz.Execute(
        output,
        policy=policy,
        nocache=True,
        available_models=available_models,
        optimization_strategy=optimization_strategy,
        execution_engine=execution_engine,
        rank=rank,
        verbose=verbose,
        allow_code_synth=allow_code_synth
    )

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
