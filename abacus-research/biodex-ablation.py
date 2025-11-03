import argparse
import json
import os
import time

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

class BiodexValidator(pz.Validator):
    def __init__(
        self,
        rp_at_k: int = 5,
        num_samples: int = 5,
        shuffle: bool = False,
        seed: int = 42,
    ):
        super().__init__()

        # read dataset and prepare entries
        dataset = datasets.load_dataset("BioDEX/BioDEX-Reactions", split="train").to_pandas()
        if shuffle:
            dataset = dataset.sample(n=num_samples, random_state=seed).to_dict(orient="records")
        else:
            dataset = dataset.to_dict(orient="records")[:num_samples]

        # compute mapping from pmid --> label (i.e. reactions list)
        self.pmid_to_label = self._compute_pmid_to_label(dataset)

        # store rp_at_k for computing rank-precision at k metric
        self.k = rp_at_k

    def _compute_pmid_to_label(self, dataset: list[dict]) -> dict:
        """Compute the label for a BioDEX report given its entry in the dataset."""
        pmid_to_label = {}
        for entry in dataset:
            pmid = str(entry["pmid"])
            reactions_lst = [
                reaction.strip().lower().replace("'", "").replace("^", "")
                for reaction in entry["reactions"].split(",")
            ]
            pmid_to_label[pmid] = reactions_lst

        return pmid_to_label

    def rank_precision_at_k(self, preds: list | None, targets: list):
        if preds is None:
            return 0.0

        try:
            # lower-case each list
            preds = [pred.strip().lower().replace("'", "").replace("^", "") for pred in preds]
            targets = set([target.strip().lower().replace("'", "").replace("^", "") for target in targets])

            # compute rank-precision at k
            rn = len(targets)
            denom = min(self.k, rn)
            total = 0.0
            for i in range(self.k):
                total += preds[i] in targets if i < len(preds) else 0.0

            return total / denom

        except Exception:
            os.makedirs("rp@k-errors", exist_ok=True)
            ts = time.time()
            with open(f"rp@k-errors/error-{ts}.txt", "w") as f:
                f.write(str(preds))
            return 0.0

    def term_recall(self, preds: list | None, targets: list):
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

    def map_score_fn(self, fields: list[str], input_record: dict, output: dict) -> float | None:
        field_name = fields[0]
        if field_name == "reactions":
            preds = output.get(field_name)
            targets = self.pmid_to_label[str(input_record["pmid"])]
            return self.term_recall(preds, targets)
        elif field_name == "ranked_reaction_labels":
            preds = output.get(field_name)
            targets = self.pmid_to_label[str(input_record["pmid"])]
            return self.rank_precision_at_k(preds, targets)
        else:
            raise NotImplementedError(f"Validator.map_score_fn not implemented for field {field_name}.")

    def topk_score_fn(self, fields: list[str], input_record: dict, output: dict) -> float | None:
        field_name = fields[0]
        if field_name == "reaction_labels":
            preds = output.get(field_name)
            targets = self.pmid_to_label[input_record["pmid"]]
            return self.term_recall(preds, targets)
        else:
            raise NotImplementedError(f"Validator.topk_score_fn not implemented for field {field_name}.")


class BiodexDataset(pz.IterDataset):
    def __init__(
        self,
        rp_at_k: int = 5,
        num_samples: int = 5,
        split: str = "test",
        shuffle: bool = False,
        seed: int = 42,
    ):
        super().__init__(id="biodex", schema=biodex_entry_cols)

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
        item = {"pmid": pmid, "title": title, "abstract": abstract, "fulltext": fulltext}

        return item


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description="Run a simple demo")
    parser.add_argument(
        "--optimizer-strategy",
        default="pareto",
        type=str,
        help="The optimizer strategy to use. One of pareto or greedy",
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
        "--policy",
        default=None,
        type=str,
        help="The policy (one of 'mincost' or 'maxquality').",
    )
    parser.add_argument(
        "--priors-file",
        default=None,
        type=str,
        help="A file with a dictionary mapping physical operator ids to prior belief on their performance",
    )
    args = parser.parse_args()

    # create directory for profiling data
    os.makedirs("ablation-data", exist_ok=True)

    seed = args.seed
    k = args.k
    j = args.j
    sample_budget = args.sample_budget
    optimizer_strategy = args.optimizer_strategy
    exp_name = args.exp_name
    priors = None
    if args.priors_file is not None and os.path.exists(args.priors_file):
        with open(args.priors_file) as f:
            priors = json.load(f)

    # set the optimization policy; constraint set to 80% of mean quality from unconstrained plans (Table 2)
    policy = (
        pz.MinCostAtFixedQuality(min_quality=0.8 * 0.261)
        if args.policy == "mincost"
        else pz.MaxQualityAtFixedCost(max_cost=0.5 * 0.7)
    )
    print(f"USING POLICY: {policy}")

    if os.getenv("OPENAI_API_KEY") is None and os.getenv("TOGETHER_API_KEY") is None and os.getenv("ANTHROPIC_API_KEY") is None:
        print("WARNING: OPENAI_API_KEY, TOGETHER_API_KEY, and ANTHROPIC_API_KEY are unset")

    # create validator
    validator = BiodexValidator(
        rp_at_k=5,
        num_samples=20,
        shuffle=True,
        seed=seed,
    )

    # create train dataset for validator
    train_dataset = BiodexDataset(
        split="train",
        num_samples=20,
        shuffle=True,
        seed=seed,
    )
    train_dataset = {train_dataset.id: train_dataset}

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
    plan = BiodexDataset(split="test", num_samples=250, shuffle=True, seed=seed)
    plan = plan.sem_map(biodex_reactions_cols)
    plan = plan.sem_topk(
        index=index,
        search_func=search_func,
        search_attr="reactions",
        output_attrs=biodex_reaction_labels_cols,
    )
    plan = plan.sem_map(biodex_ranked_reactions_labels_cols, depends_on=["title", "abstract", "fulltext", "reaction_labels"])

    # set models
    models = [
        Model.GPT_4o,
        Model.GPT_4o_MINI,
        Model.LLAMA3_1_8B,
        Model.LLAMA3_3_70B,
        # Model.MIXTRAL,  # NOTE: only available in tag `abacus-paper-experiments`
        # Model.DEEPSEEK_R1_DISTILL_QWEN_1_5B,
    ]

    # execute pz plan
    config = pz.QueryProcessorConfig(
        policy=policy,
        optimizer_strategy=optimizer_strategy,
        execution_strategy="parallel",
        use_final_op_quality=True,
        max_workers=64,
        available_models=models,
        allow_bonded_query=True,
        allow_critic=True,
        allow_mixtures=True,
        allow_rag_reduction=True,
        progress=True,
        k=k,
        j=j,
        sample_budget=sample_budget,
        # sample_cost_budget=0.10,
        seed=seed,
        exp_name=exp_name,
        priors=priors,
        dont_use_priors=(priors is None),
    )

    data_record_collection = plan.optimize_and_run(config=config, train_dataset=train_dataset, validator=validator)

    print(data_record_collection.to_df())
    data_record_collection.to_df().to_csv(f"ablation-data/{exp_name}-output.csv", index=False)

    # create filepaths for records and stats
    records_path = f"ablation-data/{exp_name}-records.json"
    stats_path = f"ablation-data/{exp_name}-profiling.json"

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
    with open(f"ablation-data/{exp_name}-metrics.json", "w") as f:
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
