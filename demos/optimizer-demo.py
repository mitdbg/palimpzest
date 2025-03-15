import argparse
import json
import os
import random
import time
from functools import partial

import chromadb
import datasets
from chromadb.utils.embedding_functions.openai_embedding_function import OpenAIEmbeddingFunction

# from ragatouille import RAGPretrainedModel
import palimpzest as pz
from palimpzest.constants import Model
from palimpzest.utils.model_helpers import get_models

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


class BiodexReader(pz.DataReader):
    def __init__(
        self,
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

        # shuffle records if shuffle = True
        if shuffle:
            random.Random(seed).shuffle(self.dataset)

        # trim to number of samples
        self.dataset = self.dataset[:num_samples]
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
            "reactions": target_reactions,
            "reaction_labels": target_reactions,
            "ranked_reaction_labels": target_reactions,
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

        # add label info
        item["labels"] = self.compute_label(entry)

        # add scoring functions for list fields
        rank_precision_at_k = partial(BiodexReader.rank_precision_at_k, k=self.rp_at_k)
        item["score_fn"]["reactions"] = BiodexReader.term_recall
        item["score_fn"]["reaction_labels"] = BiodexReader.term_recall
        item["score_fn"]["ranked_reaction_labels"] = rank_precision_at_k

        return item


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description="Run a simple demo")
    parser.add_argument("--verbose", default=False, action="store_true", help="Print verbose output")
    parser.add_argument("--progress", default=False, action="store_true", help="Print progress output")
    parser.add_argument(
        "--processing-strategy",
        default="sentinel",
        type=str,
        help="The engine to use. One of sentinel or no_sentinel",
    )
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
    parser.add_argument(
        "--exp-name",
        default=None,
        type=str,
        help="Name of experiment which is used in output filename",
    )

    args = parser.parse_args()

    # The user has to indicate the workload and experiment name
    if args.exp_name is None:
        print("Please provide an experiment name using --exp-name")
        exit(1)

    # create directory for profiling data
    os.makedirs("opt-profiling-data", exist_ok=True)

    verbose = args.verbose
    progress = args.progress
    seed = args.seed
    val_examples = args.val_examples
    k = args.k
    j = args.j
    sample_budget = args.sample_budget
    exp_name = args.exp_name
    processing_strategy = args.processing_strategy
    execution_strategy = args.execution_strategy
    sentinel_execution_strategy = args.sentinel_execution_strategy

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
        split="test",
        num_samples=250,
        shuffle=False,
        seed=seed,
    )

    # create validation data source
    val_datasource = BiodexReader(
        split="train",
        num_samples=val_examples,
        shuffle=False,
        seed=seed,
    )

    # # load index [Colbert]
    # index_path = ".ragatouille/colbert/indexes/reaction-terms"
    # index = RAGPretrainedModel.from_index(index_path)

    # def search_func(index, query, k):
    #     results = index.search(query, k=1)
    #     results = [result[0] if isinstance(result, list) else result for result in results]
    #     sorted_results = sorted(results, key=lambda result: result["score"], reverse=True)
    #     return [result["content"] for result in sorted_results[:k]], GenerationStats(model_name="colbert")

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
        return final_sorted_results[:k]

    # construct plan
    plan = pz.Dataset(datareader)
    plan = plan.sem_add_columns(biodex_reactions_cols)
    plan = plan.retrieve(
        index=index,
        search_func=search_func,
        search_attr="reactions",
        output_attr="reaction_labels",
        output_attr_desc="Most relevant official terms for adverse reactions for the provided `reactions`",
    )
    plan = plan.sem_add_columns(biodex_ranked_reactions_labels_cols, depends_on=["title", "abstract", "fulltext", "reaction_labels"])

    # only use final op quality
    use_final_op_quality = True

    # fetch available models
    available_models = get_models(include_vision=True)

    # execute pz plan
    config = pz.QueryProcessorConfig(
        policy=policy,
        cache=False,
        val_datasource=val_datasource,
        processing_strategy=processing_strategy,
        optimizer_strategy="pareto",
        sentinel_execution_strategy=sentinel_execution_strategy,
        execution_strategy=execution_strategy,
        use_final_op_quality=use_final_op_quality,
        max_workers=20,
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
        allow_split_merge=False,
        progress=progress,
    )

    data_record_collection = plan.run(
        config=config,
        k=k,
        j=j,
        sample_budget=sample_budget,
        seed=seed,
        exp_name=exp_name,
    )

    print(data_record_collection.to_df())
    data_record_collection.to_df().to_csv(f"opt-profiling-data/biodex-reactions-{exp_name}-output.csv", index=False)

    # create filepaths for records and stats
    records_path = (
        f"opt-profiling-data/biodex-reactions-{exp_name}-records.json"
        if processing_strategy in ["mab_sentinel", "random_sampling"]
        else f"opt-profiling-data/biodex-reactions-{exp_name}-records.json"
    )
    stats_path = (
        f"opt-profiling-data/biodex-reactions-{exp_name}-profiling.json"
        if processing_strategy in ["mab_sentinel", "random_sampling"]
        else f"opt-profiling-data/biodex-reactions-{exp_name}-profiling.json"
    )

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
