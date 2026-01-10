"""
NOTE: this script worked with the tag `abacus-paper-experiments` but is no longer compatible with the main branch.
"""
import argparse
import json
import os
import time

import datasets

# from ragatouille import RAGPretrainedModel
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

class BiodexDataset(pz.IterDataset):
    def __init__(
        self,
        rp_at_k: int = 5,
        num_samples: int = 5,
        split: str = "test",
        shuffle: bool = False,
        seed: int = 42,
    ):
        super().__init__(id=f"biodex-{split}", schema=biodex_entry_cols)

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
        label_dict = {"reactions": reactions_lst}
        return label_dict

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
            item["score_fn"]["reactions"] = BiodexDataset.term_recall

        return item


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description="Run a simple demo")
    parser.add_argument("--verbose", default=False, action="store_true", help="Print verbose output")
    parser.add_argument("--progress", default=True, action="store_true", help="Print progress output")
    args = parser.parse_args()

    # create directory for profiling data
    os.makedirs("priors-data", exist_ok=True)

    verbose = args.verbose
    progress = args.progress
    seed = 123 # NOTE: unique to cascades run
    execution_strategy = "parallel"
    sentinel_execution_strategy = "all"
    optimizer_strategy = "pareto"
    exp_name = f"biodex-priors-{optimizer_strategy}-seed{seed}-cascades" # NOTE: unique to cascades run

    if os.getenv("OPENAI_API_KEY") is None and os.getenv("TOGETHER_API_KEY") is None and os.getenv("ANTHROPIC_API_KEY") is None:
        print("WARNING: OPENAI_API_KEY, TOGETHER_API_KEY, and ANTHROPIC_API_KEY are unset")

    # create data source
    dataset = BiodexDataset(
        split="test",
        num_samples=1,
        shuffle=True,
        seed=seed,
    )

    # create validation data source
    train_dataset = BiodexDataset(
        split="train",
        num_samples=5,
        shuffle=True,
        seed=seed,
    )

    # construct plan
    plan = dataset
    plan = plan.sem_add_columns(biodex_reactions_cols)

    # only use final op quality
    use_final_op_quality = True

    # execute pz plan
    config = pz.QueryProcessorConfig(
        optimizer_strategy=optimizer_strategy,
        sentinel_execution_strategy=sentinel_execution_strategy,
        execution_strategy=execution_strategy,
        use_final_op_quality=use_final_op_quality,
        max_workers=64,
        verbose=verbose,
        available_models=[ # NOTE: unique to cascades run
            # Model.GPT_4o,
            Model.GPT_4o_MINI,
            Model.LLAMA3_2_3B,
            Model.LLAMA3_1_8B,
            Model.LLAMA3_3_70B,
            Model.LLAMA3_2_90B_V,
            # Model.MIXTRAL, # NOTE: only available in tag `abacus-paper-experiments`
            # Model.DEEPSEEK_V3,
            Model.DEEPSEEK_R1_DISTILL_QWEN_1_5B,
        ],
        allow_bonded_query=True,
        allow_critic=True,
        allow_mixtures=True,
        allow_rag_reduction=True,
        progress=progress,
        k=-1,
        j=-1,
        sample_budget=5*1014,
        seed=seed,
        exp_name=exp_name,
    )

    data_record_collection = plan.run(config=config, train_dataset=train_dataset, validator=pz.Validator())

    print(data_record_collection.to_df())
    data_record_collection.to_df().to_csv(f"priors-data/{exp_name}-output.csv", index=False)

    # create filepaths for records and stats
    records_path = f"priors-data/{exp_name}-records.json"
    stats_path = f"priors-data/{exp_name}-profiling.json"

    # save record outputs
    record_jsons = []
    for record in data_record_collection:
        record_dict = record.to_dict()
        record_dict = {
            k: v
            for k, v in record_dict.items()
            if k in ["pmid", "reactions"]
        }
        record_jsons.append(record_dict)

    with open(records_path, "w") as f:
        json.dump(record_jsons, f)

    # save statistics
    execution_stats_dict = data_record_collection.execution_stats.to_json()
    with open(stats_path, "w") as f:
        json.dump(execution_stats_dict, f)
