import argparse
import json
import os
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
]

biodex_reactions_cols = [
    {"name": "reactions", "type": list[str], "desc": "The list of all medical conditions experienced by the patient as discussed in the report. Try to provide as many relevant medical conditions as possible."},
]

biodex_reaction_labels_cols = [
    {"name": "reaction_labels", "type": list[str], "desc": "Official terms for medical conditions listed in `reactions`"},
]

biodex_ranked_reactions_labels_cols = [
    {"name": "ranked_reaction_labels", "type": list[str], "desc": "The ranked list of medical conditions experienced by the patient. The most relevant label occurs first in the list. Be sure to rank ALL of the inputs."},
]


class BiodexDataset(pz.IterDataset):
    def __init__(
        self,
        rp_at_k: int = 5,
        num_samples: int = 5,
        split: str = "test",
        shuffle: bool = False,
        seed: int = 42,
    ):
        super().__init__(id=f"biodex-{split}", schema=biodex_entry_cols) # TODO: this will raise a warning b/c "biodex-test" will not match "biodex-train"

        self.dataset = datasets.load_dataset("BioDEX/BioDEX-Reactions", split=split).to_pandas()
        if shuffle:
            self.dataset = self.dataset.sample(n=num_samples, random_state=seed).to_dict(orient="records")
        else:
            self.dataset = self.dataset.to_dict(orient="records")[:num_samples]

        self.rp_at_k = rp_at_k
        self.num_samples = num_samples
        self.shuffle = shuffle
        self.seed = seed
        self.split = split

    def compute_label(self, entry: dict) -> dict:
        """Compute the label for a BioDEX report given its entry in the dataset."""
        reactions_lst = [
            reaction.strip().lower().replace("'", "").replace("^", "")
            for reaction in entry["reactions"].split(",")
        ]
        label_dict = {
            "reactions": reactions_lst,
            "reaction_labels": reactions_lst,
            "ranked_reaction_labels": reactions_lst,
        }
        return label_dict

    @staticmethod
    def rank_precision_at_k(preds: list | None, targets: list, k: int):
        if preds is None:
            return 0.0

        try:
            # lower-case each list
            preds = [pred.strip().lower().replace("'", "").replace("^", "") for pred in preds]
            targets = set([target.strip().lower().replace("'", "").replace("^", "") for target in targets])

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
    def term_recall(preds: list | None, targets: list):
        if preds is None:
            return 0.0

        try:
            # normalize terms in each list
            pred_terms = set([
                term.strip()
                for pred in preds
                for term in pred.lower().replace("'", "").replace("^", "").split(" ")
            ])
            target_terms = ([
                term.strip()
                for target in targets
                for term in target.lower().replace("'", "").replace("^", "").split(" ")
            ])

            # compute term recall and return
            intersect = pred_terms.intersection(target_terms)
            term_recall = len(intersect) / len(target_terms)

            return term_recall

        except Exception:
            os.makedirs("term-recall-eval-errors", exist_ok=True)
            ts = time.time()
            with open(f"term-recall-eval-errors/error-{ts}.txt", "w") as f:
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

        # create item with fields
        item = {"fields": {}, "labels": {}, "score_fn": {}}
        item["fields"]["pmid"] = pmid
        item["fields"]["title"] = title
        item["fields"]["abstract"] = abstract
        item["fields"]["fulltext"] = fulltext

        if self.split == "train":
            # add label info
            item["labels"] = self.compute_label(entry)

            # add scoring functions for list fields
            rank_precision_at_k = partial(BiodexDataset.rank_precision_at_k, k=self.rp_at_k)
            item["score_fn"]["reactions"] = BiodexDataset.term_recall
            item["score_fn"]["reaction_labels"] = BiodexDataset.term_recall
            item["score_fn"]["ranked_reaction_labels"] = rank_precision_at_k

        return item


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description="Run a simple demo")
    parser.add_argument("--verbose", default=False, action="store_true", help="Print verbose output")
    parser.add_argument("--progress", default=False, action="store_true", help="Print progress output")
    parser.add_argument("--constrained", default=False, action="store_true", help="Use constrained objective")
    parser.add_argument("--gpt4-mini-only", default=False, action="store_true", help="Use only GPT-4o-mini")
    parser.add_argument(
        "--execution-strategy",
        default="parallel",
        type=str,
        help="The plan executor to use. One of sequential, pipelined, parallel",
    )
    parser.add_argument(
        "--sentinel-execution-strategy",
        default="mab",
        type=str,
        help="The sentinel execution strategy to use. One of mab or random",
    )
    parser.add_argument(
        "--policy",
        default="maxquality",
        type=str,
        help="One of 'mincost', 'mintime', 'maxquality'",
    )
    parser.add_argument(
        "--val-examples",
        default=25,
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
    parser.add_argument(
        "--exp-name",
        default=None,
        type=str,
        help="The experiment name.",
    )
    parser.add_argument(
        "--priors-file",
        default=None,
        type=str,
        help="A file with a dictionary mapping physical operator ids to prior belief on their performance",
    )
    parser.add_argument(
        "--quality",
        default=None,
        type=float,
        help="Quality threshold",
    )

    args = parser.parse_args()

    # create directory for profiling data
    os.makedirs("opt-profiling-data", exist_ok=True)

    verbose = args.verbose
    progress = args.progress
    seed = args.seed
    val_examples = args.val_examples
    k = args.k
    j = args.j
    sample_budget = args.sample_budget
    execution_strategy = args.execution_strategy
    sentinel_execution_strategy = args.sentinel_execution_strategy
    exp_name = (
        f"biodex-final-{sentinel_execution_strategy}-k{k}-j{j}-budget{sample_budget}-seed{seed}"
        if args.exp_name is None
        else args.exp_name
    )
    priors = None
    if args.priors_file is not None:
        with open(args.priors_file) as f:
            priors = json.load(f)

    # set the optimization policy; constraint set to 25% percentile from unconstrained plans
    policy = pz.MaxQuality() if not args.constrained else pz.MaxQualityAtFixedCost(max_cost=2.250)
    if args.quality is not None and args.policy == "mincostatfixedquality":
        policy = pz.MinCostAtFixedQuality(min_quality=args.quality)
    elif args.quality is not None and args.policy == "minlatencyatfixedquality":
        policy = pz.MinTimeAtFixedQuality(min_quality=args.quality)
    print(f"USING POLICY: {policy}")

    if os.getenv("OPENAI_API_KEY") is None and os.getenv("TOGETHER_API_KEY") is None and os.getenv("ANTHROPIC_API_KEY") is None:
        print("WARNING: OPENAI_API_KEY, TOGETHER_API_KEY, and ANTHROPIC_API_KEY are unset")

    # create data source
    dataset = BiodexDataset(
        split="test",
        num_samples=250,
        shuffle=True,
        seed=seed,
    )

    # create validation data source
    train_dataset = BiodexDataset(
        split="train",
        num_samples=val_examples,
        shuffle=True,
        seed=seed,
    )

    # load index [text-embedding-3-small]
    chroma_client = chromadb.PersistentClient(".chroma-biodex")
    openai_ef = OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"],
        model_name="text-embedding-3-small",
    )
    index = chroma_client.get_collection("biodex-reaction-terms", embedding_function=openai_ef)

    def search_func(index: chromadb.Collection, query: list[list[float]], k: int) -> list[str]:
        # execute query with embeddings
        results = index.query(query, n_results=5)

        # get list of result terms with their cosine similarity scores
        final_results = []
        for query_docs, query_distances in zip(results["documents"], results["distances"]):
            for doc, dist in zip(query_docs, query_distances):
                cosine_similarity = 1 - dist
                final_results.append({"content": doc, "similarity": cosine_similarity})

        # sort the results by similarity score
        sorted_results = sorted(final_results, key=lambda result: result["similarity"], reverse=True)

        # remove duplicates
        sorted_results_set = set()
        final_sorted_results = []
        for result in sorted_results:
            if result["content"] not in sorted_results_set:
                sorted_results_set.add(result["content"])
                final_sorted_results.append(result["content"])

        # return the top-k similar results and generation stats
        return {"reaction_labels": final_sorted_results[:k]}

    # construct plan
    plan = dataset.sem_add_columns(biodex_reactions_cols)
    plan = plan.retrieve(
        index=index,
        search_func=search_func,
        search_attr="reactions",
        output_attrs=biodex_reaction_labels_cols,
    )
    plan = plan.sem_add_columns(biodex_ranked_reactions_labels_cols, depends_on=["title", "abstract", "fulltext", "reaction_labels"])

    # set models
    models = [Model.GPT_4o_MINI] if args.gpt4_mini_only else [
        Model.GPT_4o,
        Model.GPT_4o_MINI,
        Model.LLAMA3_1_8B,
        Model.LLAMA3_3_70B,
        Model.MIXTRAL,
        Model.DEEPSEEK_R1_DISTILL_QWEN_1_5B,
    ]

    # execute pz plan
    config = pz.QueryProcessorConfig(
        policy=policy,
        optimizer_strategy="pareto",
        sentinel_execution_strategy=sentinel_execution_strategy,
        execution_strategy=execution_strategy,
        use_final_op_quality=True,
        max_workers=64,
        verbose=verbose,
        available_models=models,
        allow_bonded_query=True,
        allow_critic=True,
        allow_mixtures=True,
        allow_rag_reduction=True,
        progress=progress,
        k=k,
        j=j,
        sample_budget=sample_budget,
        seed=seed,
        exp_name=exp_name,
        priors=priors,
    )

    data_record_collection = plan.optimize_and_run(config=config, train_dataset=train_dataset, validator=pz.Validator())

    print(data_record_collection.to_df())
    data_record_collection.to_df().to_csv(f"opt-profiling-data/{exp_name}-output.csv", index=False)

    # create filepaths for records and stats
    records_path = f"opt-profiling-data/{exp_name}-records.json"
    stats_path = f"opt-profiling-data/{exp_name}-profiling.json"

    # save record outputs
    record_jsons = []
    for record in data_record_collection:
        record_dict = record.to_dict()
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

    # score output
    test_dataset = datasets.load_dataset("BioDEX/BioDEX-Reactions", split="test").to_pandas()
    test_dataset = test_dataset.sample(n=250, random_state=seed).to_dict(orient="records")

    # construct mapping from pmid --> label (field, value) pairs
    def compute_target_record(entry):
        reactions_lst = [
            reaction.strip().lower().replace("'", "").replace("^", "")
            for reaction in entry["reactions"].split(",")
        ]
        label_dict = {"ranked_reaction_labels": reactions_lst}
        return label_dict

    label_fields_to_values = {
        entry["pmid"]: compute_target_record(entry) for entry in test_dataset
    }

    def rank_precision_at_k(preds: list, targets: list, k: int):
        if preds is None:
            return 0.0

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

    def compute_avg_rp_at_k(records, k=5):
        total_rp_at_k = 0
        bad = 0
        for record in records:
            pmid = record['pmid']
            preds = record['ranked_reaction_labels']
            targets = label_fields_to_values[pmid]['ranked_reaction_labels']
            try:
                total_rp_at_k += rank_precision_at_k(preds, targets, k)
            except Exception:
                bad += 1

        return total_rp_at_k / len(records), bad

    rp_at_k, bad = compute_avg_rp_at_k(record_jsons, k=5)
    final_plan_id = list(data_record_collection.execution_stats.plan_stats.keys())[0]
    final_plan_str = data_record_collection.execution_stats.plan_strs[final_plan_id]
    stats_dict = {
        "rp@5": rp_at_k,
        "optimization_time": data_record_collection.execution_stats.optimization_time,
        "optimization_cost": data_record_collection.execution_stats.optimization_cost,
        "plan_execution_time": data_record_collection.execution_stats.plan_execution_time,
        "plan_execution_cost": data_record_collection.execution_stats.plan_execution_cost,
        "total_execution_time": data_record_collection.execution_stats.total_execution_time,
        "total_execution_cost": data_record_collection.execution_stats.total_execution_cost,
        "plan_str": final_plan_str,
    }
    with open(f"opt-profiling-data/{exp_name}-metrics.json", "w") as f:
        json.dump(stats_dict, f)

    print(f"bad: {bad}")
    print("-------")
    print(f"rp@k: {rp_at_k:.5f}")
    print(f"Optimization time: {data_record_collection.execution_stats.optimization_time}")
    print(f"Optimization cost: {data_record_collection.execution_stats.optimization_cost}")
    print(f"Plan Exec. time: {data_record_collection.execution_stats.plan_execution_time}")
    print(f"Plan Exec. cost: {data_record_collection.execution_stats.plan_execution_cost}")
    print(f"Total Execution time: {data_record_collection.execution_stats.total_execution_time}")
    print(f"Total Execution Cost: {data_record_collection.execution_stats.total_execution_cost}")
