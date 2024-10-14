import palimpzest as pz
import json
import os

from palimpzest.constants import Model
from palimpzest.elements import DataRecord
from palimpzest.utils import getModels

from ragatouille import RAGPretrainedModel
from functools import partial


class FeverClaimsSchema(pz.Schema):
    claim = pz.StringField(desc="the claim being made")
    
class FeverIntermediateSchema(pz.Schema):
    claim = pz.StringField(desc="the claim being made")
    file1 = pz.StringField(desc="most relevant wikipedia article")
    file2 = pz.StringField(desc="second most relevant wikipedia article")
    file3 = pz.StringField(desc="third most relevant wikipedia article")

class FeverOutputSchema(pz.Schema):
    label = pz.BooleanField("Output TRUE if the `claim` is supported by the evidence in `file1`, `file2`, and `file3`; output FALSE otherwise.")

def get_relevant_content(index, k, record):
    # input: record of type FeverInputSchema
    # output: record of type FeverIntermediateSchema
    relevant_files = index.search(record.claim , k=k)
    most_relevant_files = [relevant_file["content"] for relevant_file in relevant_files]
   
    # create output DataRecord
    out_record = pz.DataRecord.fromParent(FeverIntermediateSchema, parent_record=record)
    
    if len(most_relevant_files) < 3:
        most_relevant_files += [""] * (3 - len(most_relevant_files))
        
    out_record.file1 = most_relevant_files[0]
    out_record.file2 = most_relevant_files[1]
    out_record.file3 = most_relevant_files[2]
    return out_record

def set_input_schema(input: DataRecord):
    # , project_cols=[]
    output = DataRecord.fromParent(FeverClaimsSchema, parent_record=input)
    output.claim = input.contents
    return output
         
         
if "DSP_CACHEBOOL" not in os.environ or os.environ["DSP_CACHEBOOL"].lower() != "false":
        raise Exception("TURN OFF DSPy CACHE BY SETTING `export DSP_CACHEBOOL=False")
    
if os.getenv("OPENAI_API_KEY") is None and os.getenv("TOGETHER_API_KEY") is None:
        print("WARNING: Both OPENAI_API_KEY and TOGETHER_API_KEY are unset")
        
# params
dataset_id = "fever-10"
workload = "fever"

index_path = ".ragatouille/colbert/indexes/fever-articles-10-index"
index = RAGPretrainedModel.from_index(index_path)
k = 3

rank=4
num_samples=5

engine = "sentinel"
executor = "sequential"
model = "gpt-4o"
policy_type = "maxquality"

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
    optimization_strategy = pz.OptimizationStrategy.OPTIMAL
    available_models = getModels(include_vision=True)
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

claims = pz.Dataset(dataset_id, schema=pz.TextFile,
                    desc="Contains the claims that need to be verified.") 
claims = claims.convert(outputSchema=FeverClaimsSchema, udf=set_input_schema)
claims_and_relevant_files = claims.convert(outputSchema=FeverIntermediateSchema,
                                           udf=partial(get_relevant_content, index, k))
output = claims_and_relevant_files.convert(outputSchema=FeverOutputSchema)

# execute pz plan

records, execution_stats = pz.Execute(
        output,
        policy=policy,
        nocache=True,
        available_models=available_models,
        optimization_strategy=optimization_strategy,
        execution_engine=execution_engine,
        verbose=True,
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
    record_dict = record._asDict()
    ### field_to_keep = ["claim", "id", "label"]
    ### record_dict = {k: v for k, v in record_dict.items() if k in fields_to_keep}
    record_jsons.append(record_dict)
    
with open(records_path, 'w') as f:
    json.dump(record_jsons, f)

# save statistics
execution_stats_dict = execution_stats.to_json()
with open(stats_path, "w") as f:
    json.dump(execution_stats_dict, f)
