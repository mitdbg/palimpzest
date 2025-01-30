import argparse
import json
import os
import random
import time
from pathlib import Path

import datasets
from ragatouille import RAGPretrainedModel

from palimpzest.constants import Model
from palimpzest.core.data.datasources import ValidationDataSource
from palimpzest.core.elements.records import DataRecord
from palimpzest.core.lib.fields import BooleanField, ImageFilepathField, ListField, NumericField, StringField
from palimpzest.core.lib.schemas import Schema, TextFile
from palimpzest.datamanager.datamanager import DataDirectory
from palimpzest.policy import MaxQuality, MinCost, MinTime
from palimpzest.query.processor.config import QueryProcessorConfig
from palimpzest.sets import Dataset
from palimpzest.utils.model_helpers import get_models


class BiodexEntry(Schema):
    """A single entry in the Biodex ICSR Dataset."""

    pmid = StringField(desc="The PubMed ID of the medical paper")
    title = StringField(desc="The title of the medical paper")
    abstract = StringField(desc="The abstract of the medical paper")
    fulltext = StringField(
        desc="The full text of the medical paper, which contains information relevant for creating a drug safety report.",
    )

class BiodexReactions(BiodexEntry):
    """
    You will be presented with the text of a medical article which is partially or entirely about
    an adverse event experienced by a patient in response to taking one or more drugs. In this task,
    you will be asked to extract a list of the primary adverse reactions which are experienced by the patient.
    """

    reactions = ListField(
        desc='The **list** of all reaction terms discussed in the report.',
        element_type=StringField,
    )


class BiodexRankedReactions(BiodexReactions):
    """
    You will be presented with the text of a medical article which is partially or entirely about
    an adverse event experienced by a patient in response to taking one or more drugs. You will also
    be presented with a list of inferred reactions, and a set of retrieved labels which were matched
    to these inferred reactions. In this task, you are asked to output a ranked list of the labels
    which are most applicable based on the context of the article. Your output list must:
    - contain only elements from `reaction_labels`
    - place the most likely label first and the least likely label last
    - you may omit labels if you think they do not describe a reaction experienced by the patient
    """

    ranked_reaction_labels = ListField(
        desc="The ranked list of labels for adverse reactions experienced by the patient. The most likely label occurs first in the list.",
        element_type=StringField,
    )


class BiodexValidationSource(ValidationDataSource):
    def __init__(
        self,
        dataset_id,
        reactions_only: bool = True,
        rp_at_k: int = 5,
        num_samples: int = 5,
        shuffle: bool = False,
        seed: int = 42,
    ):
        super().__init__(BiodexEntry, dataset_id)
        self.dataset = datasets.load_dataset("BioDEX/BioDEX-ICSR")
        self.train_dataset = [self.dataset["train"][idx] for idx in range(250)]

        # sample from full test dataset
        self.test_dataset = [self.dataset["test"][idx] for idx in range(len(self.dataset["test"]))]
        self.test_dataset = self.test_dataset[:250]  # use first 250 to compare directly with biodex

        self.reactions_only = reactions_only
        self.rp_at_k = rp_at_k
        self.num_samples = num_samples
        self.shuffle = shuffle
        self.seed = seed

        # construct mapping from listing --> label (field, value) pairs
        def compute_target_record(entry, reactions_only: bool = False):
            target_lst = entry["target"].split("\n")
            label_dict = {
                "serious": int(target_lst[0].split(":")[-1]),
                "patientsex": int(target_lst[1].split(":")[-1]),
                "drugs": [drug.strip().lower() for drug in target_lst[2].split(":")[-1].split(",")],
                "reactions": [reaction.strip().lower() for reaction in target_lst[3].split(":")[-1].split(",")],
                "reaction_labels": [reaction.strip().lower() for reaction in target_lst[3].split(":")[-1].split(",")],
                "ranked_reaction_labels": [
                    reaction.strip().lower() for reaction in target_lst[3].split(":")[-1].split(",")
                ],
            }
            if reactions_only:
                label_dict = {
                    k: v
                    for k, v in label_dict.items()
                    if k in ["reactions", "reaction_labels", "ranked_reaction_labels"]
                }
            return label_dict

        self.label_fields_to_values = {
            entry["pmid"]: compute_target_record(entry, reactions_only=reactions_only) for entry in self.train_dataset
        }

        # shuffle records if shuffle = True
        if shuffle:
            random.Random(seed).shuffle(self.train_dataset)

        # trim to number of samples
        self.train_dataset = self.train_dataset[:num_samples]

    def copy(self):
        return BiodexValidationSource(self.dataset_id, self.num_samples, self.shuffle, self.seed)

    def __len__(self):
        return len(self.test_dataset)

    def get_val_length(self):
        return len(self.train_dataset)

    def get_size(self):
        return 0

    def get_field_to_metric_fn(self):
        # define f1 function
        def f1_eval(preds: list, targets: list):
            if preds is None:
                return 0.0

            try:
                # compute precision and recall
                s_preds = set([pred.lower() for pred in preds])
                s_targets = set([target.lower() for target in targets])

                intersect = s_preds.intersection(s_targets)

                precision = len(intersect) / len(s_preds) if len(s_preds) > 0 else 0.0
                recall = len(intersect) / len(s_targets)

                # compute f1 score and return
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

                return f1

            except Exception:
                os.makedirs("f1-errors", exist_ok=True)
                ts = time.time()
                with open(f"f1-errors/error-{ts}.txt", "w") as f:
                    f.write(str(preds))
                return 0.0

        # define rank precision at k
        def rank_precision_at_k(preds: list, targets: list):
            if preds is None:
                return 0.0

            try:
                # lower-case each list
                preds = [pred.lower() for pred in preds]
                targets = set([target.lower() for target in targets])

                # compute rank-precision at k
                rn = len(targets)
                denom = min(self.rp_at_k, rn)
                total = 0.0
                for i in range(self.rp_at_k):
                    total += preds[i] in targets if i < len(preds) else 0.0

                return total / denom

            except Exception:
                os.makedirs("rp@k-errors", exist_ok=True)
                ts = time.time()
                with open(f"rp@k-errors/error-{ts}.txt", "w") as f:
                    f.write(str(preds))
                return 0.0

        # define quality eval function for drugs and reactions fields
        fields_to_metric_fn = {}
        if self.reactions_only:
            fields_to_metric_fn = {
                "reactions": f1_eval,
                "reaction_labels": f1_eval,
                "ranked_reaction_labels": rank_precision_at_k,
            }

        else:
            fields_to_metric_fn = {
                "serious": "exact",
                "patientsex": "exact",
                "drugs": f1_eval,
                "reactions": f1_eval,
            }

        return fields_to_metric_fn

    def get_item(self, idx: int, val: bool = False, include_label: bool = False):
        # fetch entry
        entry = self.test_dataset[idx] if not val else self.train_dataset[idx]

        # create data record
        dr = DataRecord(self.schema, source_id=entry["pmid"])
        dr.pmid = entry["pmid"]
        dr.title = entry["title"]
        dr.abstract = entry["abstract"]
        dr.fulltext = entry["fulltext"]

        # if requested, also return the label information
        if include_label:
            # augment data record with label info
            labels_dict = self.label_fields_to_values[entry["pmid"]]

            for field, value in labels_dict.items():
                setattr(dr, field, value)

        return dr


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description="Run a simple demo")
    parser.add_argument("--verbose", default=False, action="store_true", help="Print verbose output")
    parser.add_argument(
        "--workload", type=str, help="The workload to run. One of enron, real-estate, biodex, biodex-reactions."
    )
    parser.add_argument(
        "--processing-strategy",
        default="mab_sentinel",
        type=str,
        help="The engine to use. One of mab_sentinel, no_sentinel, random_sampling",
    )
    parser.add_argument(
        "--execution-strategy",
        default="pipelined_parallel",
        type=str,
        help="The plan executor to use. One of sequential, pipelined_single_thread, pipelined_parallel",
    )
    parser.add_argument(
        "--policy",
        default="mincost",
        type=str,
        help="One of 'mincost', 'mintime', 'maxquality'",
    )
    parser.add_argument(
        "--val-examples",
        default=5,
        type=int,
        help="Number of validation examples to sample from",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        type=str,
        help="One of 'gpt-4o', 'gpt-4o-mini', 'llama', 'mixtral'",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Seed used to initialize RNG for MAB sampling algorithm",
    )
    parser.add_argument(
        "--k",
        default=10,
        type=int,
        help="Number of columns to sample in Random Sampling or MAB sentinel execution",
    )
    parser.add_argument(
        "--j",
        default=3,
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
    parser.add_argument(
        "--sample-start-idx",
        default=None,
        type=int,
        help="",
    )
    parser.add_argument(
        "--sample-end-idx",
        default=None,
        type=int,
        help="",
    )
    parser.add_argument(
        "--exp-name",
        default=None,
        type=str,
        help="Name of experiment which is used in output filename",
    )

    args = parser.parse_args()

    # The user has to indicate the dataset id and the workload
    if args.workload is None:
        print("Please provide a workload using --workload")
        exit(1)
    if args.exp_name is None:
        print("Please provide an experiment name using --exp-name")
        exit(1)

    # create directory for profiling data
    os.makedirs("opt-profiling-data", exist_ok=True)
    workload = args.workload
    verbose = args.verbose
    seed = args.seed
    val_examples = args.val_examples
    k = args.k
    j = args.j
    sample_budget = args.sample_budget
    sample_all_ops = args.sample_all_ops
    sample_all_records = args.sample_all_records
    sample_start_idx = args.sample_start_idx
    sample_end_idx = args.sample_end_idx
    exp_name = args.exp_name

    policy = MaxQuality()
    if args.policy == "mincost":
        policy = MinCost()
    elif args.policy == "mintime":
        policy = MinTime()
    elif args.policy == "maxquality":
        policy = MaxQuality()
    else:
        print("Policy not supported for this demo")
        exit(1)

    if os.getenv("OPENAI_API_KEY") is None and os.getenv("TOGETHER_API_KEY") is None:
        print("WARNING: Both OPENAI_API_KEY and TOGETHER_API_KEY are unset")

    # create pz plan
    plan, use_final_op_quality = None, False
    if workload == "biodex-reactions":
        user_dataset_id = "biodex-user"

        # load index
        index_path = ".ragatouille/colbert/indexes/reaction-terms"
        index = RAGPretrainedModel.from_index(index_path)

        # create and register validation data source
        datasource = BiodexValidationSource(
            dataset_id=f"{user_dataset_id}",
            num_samples=val_examples,
            shuffle=False,
            seed=seed,
        )
        DataDirectory().register_user_source(
            src=datasource,
            dataset_id=f"{user_dataset_id}",
        )
        plan = Dataset(user_dataset_id, schema=BiodexEntry)
        plan = plan.convert(BiodexReactions)  # infer

        def search_func(index, query, k):
            results = index.search(query, k=1)
            results = [result[0] if isinstance(result, list) else result for result in results]
            sorted_results = sorted(results, key=lambda result: result["score"], reverse=True)
            return [result["content"] for result in sorted_results[:k]]

        plan = plan.retrieve(
            index=index,
            search_func=search_func,
            search_attr="reactions",
            output_attr="reaction_labels",
            output_attr_desc="Most relevant official terms for adverse reactions for the provided `reactions`",
            # k=10, # if we set k, then it will be fixed; if we leave it unspecified then the optimizer will choose
        )  # TODO: retrieve (top-1 retrieve per prediction? or top-k retrieve for all predictions?)
        plan = plan.convert(BiodexRankedReactions)

        # only use final op quality
        use_final_op_quality = True

    elif workload == "biodex":
        pass

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
            "deepseek": Model.DEEPSEEK,
            "llama": Model.LLAMA3,
        }
        model_str_to_vision_model = {
            "gpt-4o": Model.GPT_4o_V,
            "gpt-4o-mini": Model.GPT_4o_MINI_V,
            "mixtral": Model.LLAMA3_V,
            "deepseek": Model.LLAMA3_V,
            "llama": Model.LLAMA3_V,
        }
        optimizer_strategy = "none"
        available_models = [model_str_to_model[args.model]] + [model_str_to_vision_model[args.model]]

    # execute pz plan
    config = QueryProcessorConfig(
        policy=policy,
        nocache=True,
        # available_models=available_models,
        processing_strategy=args.processing_strategy,
        optimizer_strategy=optimizer_strategy,
        execution_strategy=args.execution_strategy,
        allow_code_synth=False,  # (workload != "biodex"),
        use_final_op_quality=use_final_op_quality,
        max_workers=1,
        verbose=verbose,
        available_models=[
            Model.GPT_4o,
            Model.GPT_4o_V,
            Model.GPT_4o_MINI,
            Model.GPT_4o_MINI_V,
            # Model.DEEPSEEK,
            Model.MIXTRAL,
            # Model.LLAMA3,
            # Model.LLAMA3_V,
        ],
    )

    data_record_collection = plan.run(
        config=config,
        k=k,
        j=j,
        sample_budget=sample_budget,
        sample_all_ops=sample_all_ops,
        sample_all_records=sample_all_records,
        sample_start_idx=sample_start_idx,
        sample_end_idx=sample_end_idx,
        seed=seed,
        exp_name=exp_name,
    )

    print(data_record_collection.to_df())

    # create filepaths for records and stats
    records_path = (
        f"opt-profiling-data/{workload}-{exp_name}-records.json"
        if args.processing_strategy in ["mab_sentinel", "random_sampling"]
        else f"opt-profiling-data/{workload}-baseline-{exp_name}-records.json"
    )
    stats_path = (
        f"opt-profiling-data/{workload}-{exp_name}-profiling.json"
        if args.processing_strategy in ["mab_sentinel", "random_sampling"]
        else f"opt-profiling-data/{workload}-baseline-{exp_name}-profiling.json"
    )

    # save record outputs
    record_jsons = []
    for record in data_record_collection:
        record_dict = record.to_dict()
        if workload == "biodex-reactions":
            record_dict = {
                k: v
                for k, v in record_dict.items()
                if k in ["pmid", "reactions", "reaction_labels", "ranked_reaction_labels"]
            }
        record_jsons.append(record_dict)

    with open(records_path, "w") as f:
        json.dump(record_jsons, f)

    # save statistics
    execution_stats_dict = data_record_collection.execution_stats.to_json()
    with open(stats_path, "w") as f:
        json.dump(execution_stats_dict, f)
