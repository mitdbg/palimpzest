import argparse
import json
import os
import random
import time
from functools import partial

import chromadb
import datasets
from chromadb.utils.embedding_functions.openai_embedding_function import OpenAIEmbeddingFunction

import palimpzest as pz
from palimpzest.constants import Model

biodex_entry_cols = [
    {"name": "pmid", "type": str, "desc": "The PubMed ID of the medical paper"},
    {"name": "title", "type": str, "desc": "The title of the medical paper"},
    {"name": "abstract", "type": str, "desc": "The abstract of the medical paper"},
    {"name": "fulltext", "type": str, "desc": "The full text of the medical paper, which contains information relevant for creating a drug safety report."},
    {"name": "conditions", "type": list[str], "desc": "A list of medical conditions."},
]

biodex_relevant_conditions_cols = [
    {"name": "relevant_conditions", "type": list[str], "desc": "The list of medical conditions which are relevant to the patient as described by the report."},
]

biodex_ranked_relevant_conditions_cols = [
    {"name": "ranked_relevant_conditions", "type": list[str], "desc": "The ranked list of medical conditions experienced by the patient. The most relevant label occurs first in the list."},
]


class BiodexReader(pz.DataReader):
    def __init__(
        self,
        reactions_only: bool = True,
        rp_at_k: int = 5,
        num_samples: int = 5,
        split: str = "test",
        shuffle: bool = False,
        seed: int = 42,
    ):
        super().__init__(biodex_entry_cols)

        # for some weird reason we need to put the dataset through a generator to get items as dicts
        self.dataset = datasets.load_dataset("BioDEX/BioDEX-ICSR")
        self.dataset = [self.dataset[split][idx] for idx in range(len(self.dataset[split]))]

        # get list of all reaction terms
        self.reaction_terms = []
        with open('testdata/reaction_terms.txt') as f:
            for line in f:
                self.reaction_terms.append(line.strip())

        # shuffle records if shuffle = True
        if shuffle:
            random.Random(seed).shuffle(self.dataset)

        # trim to number of samples
        self.dataset = self.dataset[:num_samples]
        self.reactions_only = reactions_only
        self.rp_at_k = rp_at_k
        self.num_samples = num_samples
        self.shuffle = shuffle
        self.seed = seed

    def compute_label(self, entry: dict) -> dict:
        """Compute the label for a BioDEX report given its entry in the dataset."""
        target_lst = entry["target"].split("\n")
        target_reactions = [
            reaction.strip().lower().replace("'", "").replace("^", "")
            for reaction in target_lst[3].split(":")[-1].split(",")
        ]
        label_dict = {
            "conditions": target_reactions,
            "relevant_conditions": target_reactions,
            "ranked_relevant_conditions": target_reactions,
        }

        return label_dict

    @staticmethod
    def rank_precision_at_k(preds: list | None, targets: list, k: int):
        if preds is None:
            return 0.0

        try:
            # lower-case each list
            preds = [pred.lower().replace("'", "").replace("^", "") for pred in preds]
            targets = set([target.lower().replace("'", "").replace("^", "") for target in targets])

            # compute rank-precision at k
            rn = len(targets)
            denom = min(k, rn)
            total = 0.0
            for i in range(k):
                total += preds[i] in targets if i < len(preds) else 0.0

            return total / denom

        except Exception:
            os.makedirs("rp@k-errors", exist_ok=True)
            ts = time.time()
            with open(f"rp@k-errors/error-{ts}.txt", "w") as f:
                f.write(str(preds))
            return 0.0

    @staticmethod
    def recall_eval(preds: list | None, targets: list):
        if preds is None:
            return 0.0

        try:
            # compute precision and recall
            s_preds = set([pred.lower().replace("'", "").replace("^", "") for pred in preds])
            s_targets = set([target.lower().replace("'", "").replace("^", "") for target in targets])

            intersect = s_preds.intersection(s_targets)

            precision = len(intersect) / len(s_preds) if len(s_preds) > 0 else 0.0
            recall = len(intersect) / len(s_targets)

            # compute f1 score and return
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            return recall

        except Exception:
            os.makedirs("recall-eval-errors", exist_ok=True)
            ts = time.time()
            with open(f"recall-eval-errors/error-{ts}.txt", "w") as f:
                f.write(str(preds))
            return 0.0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        # get entry
        entry = self.dataset[idx]

        # get input fields
        pmid = entry["pmid"]
        title = entry["title"]
        abstract = entry["abstract"]
        fulltext = entry["fulltext"]

        # iterate through reaction terms and return all terms that have every word in fulltext
        conditions = []
        fulltext_words = set(word.lower().replace("'","").replace("^","").strip() for word in fulltext.split(" "))
        for term in self.reaction_terms:
            if all(word.lower().replace("'", "").replace("^","").strip() in fulltext_words for word in term.split(" ")):
                conditions.append(term)

        # create item with fields
        item = {"fields": {}, "labels": {}, "score_fn": {}}
        item["fields"]["pmid"] = pmid
        item["fields"]["title"] = title
        item["fields"]["abstract"] = abstract
        item["fields"]["fulltext"] = fulltext
        item["fields"]["conditions"] = conditions

        # add label info
        item["labels"] = self.compute_label(entry)

        # add scoring functions for list fields
        rank_precision_at_k = partial(BiodexReader.rank_precision_at_k, k=self.rp_at_k)
        item["score_fn"]["conditions"] = BiodexReader.recall_eval
        item["score_fn"]["relevant_conditions"] = BiodexReader.recall_eval
        item["score_fn"]["ranked_relevant_conditions"] = rank_precision_at_k

        return item


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

    # The user has to indicate the workload and experiment name
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

    if os.getenv("OPENAI_API_KEY") is None and os.getenv("TOGETHER_API_KEY") is None:
        print("WARNING: Both OPENAI_API_KEY and TOGETHER_API_KEY are unset")

    # create data source
    datareader = BiodexReader(
        reactions_only=True,
        split="test",
        num_samples=250,
        shuffle=False,
        seed=seed,
    )

    # create validation data source
    val_datasource = BiodexReader(
        reactions_only=True,
        split="train",
        num_samples=val_examples,
        shuffle=False,
        seed=seed,
    )

    # construct plan
    plan = pz.Dataset(datareader)
    plan = plan.sem_add_columns(biodex_relevant_conditions_cols)
    plan = plan.sem_add_columns(biodex_ranked_relevant_conditions_cols, depends_on=["title", "abstract", "fulltext", "relevant_conditions"])

    # only use final op quality
    use_final_op_quality = True

    # execute pz plan
    config = pz.QueryProcessorConfig(
        policy=policy,
        cache=False,
        val_datasource=val_datasource,
        processing_strategy=args.processing_strategy,
        optimizer_strategy="pareto",
        execution_strategy=args.execution_strategy,
        use_final_op_quality=use_final_op_quality,
        max_workers=1,
        verbose=verbose,
        available_models=[
            # Model.GPT_4o,
            # Model.GPT_4o_V,
            Model.GPT_4o_MINI,
            # Model.GPT_4o_MINI_V,
            # Model.DEEPSEEK,
            # Model.MIXTRAL,
            # Model.LLAMA3,
            # Model.LLAMA3_V,
        ],
        allow_bonded_query=True,
        allow_code_synth=False,
        allow_critic=True,
        allow_mixtures=True,
        allow_rag_reduction=True,
        allow_token_reduction=False,
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
    data_record_collection.to_df().to_csv(f"opt-profiling-data/{workload}-{exp_name}-output.csv", index=False)

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
        record_dict = {
            k: v
            for k, v in record_dict.items()
            if k in ["pmid", "conditions", "relevant_conditions", "ranked_relevant_conditions"]
        }
        record_jsons.append(record_dict)

    with open(records_path, "w") as f:
        json.dump(record_jsons, f)

    # save statistics
    execution_stats_dict = data_record_collection.execution_stats.to_json()
    with open(stats_path, "w") as f:
        json.dump(execution_stats_dict, f)
