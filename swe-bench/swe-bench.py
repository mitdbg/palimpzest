import palimpzest as pz
from palimpzest.core.elements.records import DataRecord 
from palimpzest.core.lib.schemas import RawJSONObject, TextFile, Schema
from palimpzest.core.lib.fields import Field, StringField 
from palimpzest.query.processor.config import QueryProcessorConfig
from palimpzest.utils.model_helpers import get_models
from palimpzest.constants import Model

import json
import os
import argparse
import random
import swebench.harness.run_evaluation as swe_bench_eval

class GithubIssue(Schema):
    """Represents a github issue containing a problem statement and the relevant issue code pertaining to the issue."""

    instance_id = StringField(
        desc="The instance_id",
    )
    problem_statement = StringField(
        desc="A text description of the github issue which can be found within the problem statement field of the provided json object. It may also include helpful exchanges between developers relating to the issue.",
    )
    base_commit = StringField(
        desc="This is the commit that the code is based on.",
    )

class GithubIssueDataReader(pz.DataReader):
    def __init__(
        self, 
        path, 
        num_samples: int = 5, 
        shuffle: bool = False, 
        split: str = "test",
        seed: int = 42,
    ):
        super().__init__(GithubIssue)
        self.split = split 
        self.path = path
        self.num_samples = num_samples

        files = [
            os.path.join(path, filename)
            for filename in sorted(os.listdir(path))
            if os.path.isfile(os.path.join(path, filename))
        ]

        if shuffle:
            files = random.Random(seed).shuffle(files)

        self.filepaths = files[:num_samples]

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx: int):
        # get instance
        instance_filepath = self.filepaths[idx]
        with open(instance_filepath) as f:
            contents = f.read()

        data = json.loads(contents)

        github_issue = {"score_fn": {}}
        github_issue['instance_id'] = data['instance_id']
        github_issue['problem_statement'] = data['problem_statement']
        github_issue['base_commit'] = data['base_commit']

        if self.split == "train":
            # add label info
            # There are no labels for swe-bench, just scoring of the patches 
            github_issue["labels"] = 'null' 

            # Scoring function
            github_issue["score_fn"]["model_patch"] = GithubIssueDataReader.compute_swe_bench_score

        return github_issue

    @staticmethod
    def compute_swe_bench_score(pred: str, target: str):
        # Write pred to input file 

        # Run evaluation 
        # swe_bench_eval.main(
        #     dataset_name="princeton-nlp/SWE-bench_Lite",
        #     predictions_path="/Users/jasonli/Documents/GitHub/palimpzest/swebench",
        #     instance_ids=[], 
        #     run_id=f"run",
        # )

        # Parse the evaluation result file 

        return 1 
        

def buildSweBenchPlan(path):
    # create instance of data reader
    datareader = GithubIssueDataReader(path, num_samples=300, split="test")

    # Create dataset
    github_issues = pz.Dataset(datareader)

    # Process GitHub Issues
    code_plans = github_issues.add_agent(agent_name="debugger")
    code_patches = code_plans.add_agent(agent_name="code_editor") 

    return code_patches

def dump_records(filename, records, values, all_values=False):
    # Filter the dictionaries to include only specified columns (keys)
    dict_list = []

    for record in records:
        if all_values: 
            instance_patch = record.to_dict()
        else: 
            instance_patch = {key: record.to_dict().get(key) for key in values if key in record.to_dict()}
        instance_patch['model_name_or_path'] = 'palimpzest'
        dict_list.append(instance_patch)

    # Create the file (or clear it if it exists)
    open(filename, 'w').close()

    # Write the filtered list of dictionaries to the file
    with open(filename, 'w') as file:
        json.dump(dict_list, file, indent=4)

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description="Run a simple demo")
    parser.add_argument("--verbose", default=False, action="store_true", help="Print verbose output")

    parser.add_argument(
        "--processing_strategy",
        default="mab_sentinel",
        type=str,
        help="The engine to use. One of mab_sentinel, no_sentinel, random_sampling",
    )
    parser.add_argument(
        "--execution_strategy",
        default="pipelined_parallel",
        type=str,
        help="The plan executor to use. One of sequential, pipelined_single_thread, pipelined_parallel",
    )
    parser.add_argument(
        "--policy",
        default="maxquality",
        type=str,
        help="One of 'mincost', 'mintime', 'maxquality'",
    )
    parser.add_argument(
        "--val_examples",
        default=2,
        type=int,
        help="Number of validation examples to sample from",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        type=str,
        help="One of 'gpt-4o', 'gpt-4o-mini'"
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Seed used to initialize RNG for MAB sampling algorithm",
    )
    parser.add_argument(
        "--k",
        default=2,
        type=int,
        help="Number of columns to sample in Random Sampling or MAB sentinel execution",
    )
    parser.add_argument(
        "--j",
        default=5,
        type=int,
        help="Number of columns to sample in Random Sampling or MAB sentinel execution",
    )
    parser.add_argument(
        "--sample-budget",
        default=100,
        type=int,
        help="Total sample budget in Random Sampling or MAB sentinel execution",
)
    parser.add_argument("--sample-all-ops", default=False, action="store_true", help="Sample all operators")
    parser.add_argument("--sample-all-records", default=False, action="store_true", help="Sample all records")

    args = parser.parse_args()

    # Select the policy
    policy = pz.MaxQuality()
    if args.policy == "mincost":
        policy = pz.MinCost()
    elif args.policy == "mintime":
        policy = pz.MinTime()
    elif args.policy == "maxquality":
        policy = pz.MaxQuality()
    else:
        print("Policy not supported for this demo")
        exit(1)

    # select optimization strategy and available models based on engine
    optimizer_strategy, available_models = None, None
    if args.processing_strategy in ["mab_sentinel", "random_sampling"]:
        optimizer_strategy = "pareto"
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
        optimizer_strategy = "none"
        available_models = [model_str_to_model[args.model]] + [model_str_to_vision_model[args.model]]

    path = "/Users/jasonli/Documents/GitHub/palimpzest/testdata/swe-bench-lite"
    plan = buildSweBenchPlan(path)

    # Create the validation data source
    val_datasource = GithubIssueDataReader(
        path="/Users/jasonli/Documents/GitHub/palimpzest/testdata/swe-bench-lite",
        num_samples=args.val_examples,
        split="train",
        # shuffle=True,
    )

    config = pz.QueryProcessorConfig(
        policy=policy,
        nocache=True,
        val_datasource=val_datasource,
        available_models=available_models,
        processing_strategy=args.processing_strategy,
        optimizer_strategy=optimizer_strategy,
        execution_strategy=args.execution_strategy,
        use_final_op_quality=True,
        # max_workers=10,
        verbose=args.verbose,
    )

    data_record_collection = plan.run(
        config=config,
        k=args.k,
        j=args.j,
        sample_budget=args.sample_budget,
        # sample_all_ops=args.sample_all_ops,
        # sample_all_records=args.sample_all_records,
        # sample_start_idx=sample_start_idx,
        # sample_end_idx=sample_end_idx,
        seed=args.seed,
        # exp_name=exp_name,
    )

    print('COMPLETE')

    # Output Records into json file
    # filename = f'output_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.json' 
    #dump_records(filename, data_record_collection, values=['instance_id', 'model_patch']) 
