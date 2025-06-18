import argparse
import json
import os
import string
from functools import partial

import datasets
import numpy as np
import pandas as pd

import palimpzest as pz
from palimpzest.constants import Model
from palimpzest.core.lib.fields import ListField, StringField

cuad_categories = [
    {
        "Category": "Document Name",
        "Description": "The name of the contract",
        "Answer Format": "Contract Name",
        "Group": "Group: -",
    },
    {
        "Category": "Parties",
        "Description": "The two or more parties who signed the contract",
        "Answer Format": "Entity or individual names",
        "Group": "Group: -",
    },
    {
        "Category": "Agreement Date",
        "Description": "The date of the contract",
        "Answer Format": "Date (mm/dd/yyyy)",
        "Group": "Group: 1",
    },
    {
        "Category": "Effective Date",
        "Description": "The date when the contract is effective\u00a0",
        "Answer Format": "Date (mm/dd/yyyy)",
        "Group": "Group: 1",
    },
    {
        "Category": "Expiration Date",
        "Description": "On what date will the contract's initial term expire?",
        "Answer Format": "Date (mm/dd/yyyy) / Perpetual",
        "Group": "Group: 1",
    },
    {
        "Category": "Renewal Term",
        "Description": "What is the renewal term after the initial term expires? This includes automatic extensions and unilateral extensions with prior notice.",
        "Answer Format": "[Successive] number of years/months / Perpetual",
        "Group": "Group: 1",
    },
    {
        "Category": "Notice Period to Terminate Renewal",
        "Description": "What is the notice period required to terminate renewal?",
        "Answer Format": "Number of days/months/year(s)",
        "Group": "Group: 1",
    },
    {
        "Category": "Governing Law",
        "Description": "Which state/country's law governs the interpretation of the contract?",
        "Answer Format": "Name of a US State / non-US Province, Country",
        "Group": "Group: -",
    },
    {
        "Category": "Most Favored Nation",
        "Description": "Is there a clause that if a third party gets better terms on the licensing or sale of technology/goods/services described in the contract, the buyer of such technology/goods/services under the contract shall be entitled to those better terms?",
        "Answer Format": "Yes/No",
        "Group": "Group: -",
    },
    {
        "Category": "Non-Compete",
        "Description": "Is there a restriction on the ability of a party to compete with the counterparty or operate in a certain geography or business or technology sector?\u00a0",
        "Answer Format": "Yes/No",
        "Group": "Group: 2",
    },
    {
        "Category": "Exclusivity",
        "Description": "Is there an exclusive dealing\u00a0 commitment with the counterparty? This includes a commitment to procure all \u201crequirements\u201d from one party of certain technology, goods, or services or a prohibition on licensing or selling technology, goods or services to third parties, or a prohibition on\u00a0 collaborating or working with other parties), whether during the contract or\u00a0 after the contract ends (or both).",
        "Answer Format": "Yes/No",
        "Group": "Group: 2",
    },
    {
        "Category": "No-Solicit of Customers",
        "Description": "Is a party restricted from contracting or soliciting customers or partners of the counterparty, whether during the contract or after the contract ends (or both)?",
        "Answer Format": "Yes/No",
        "Group": "Group: 2",
    },
    {
        "Category": "Competitive Restriction Exception",
        "Description": "This category includes the exceptions or carveouts to Non-Compete, Exclusivity and No-Solicit of Customers above.",
        "Answer Format": "Yes/No",
        "Group": "Group: 2",
    },
    {
        "Category": "No-Solicit of Employees",
        "Description": "Is there a restriction on a party\u2019s soliciting or hiring employees and/or contractors from the\u00a0 counterparty, whether during the contract or after the contract ends (or both)?",
        "Answer Format": "Yes/No",
        "Group": "Group: -",
    },
    {
        "Category": "Non-Disparagement",
        "Description": "Is there a requirement on a party not to disparage the counterparty?",
        "Answer Format": "Yes/No",
        "Group": "Group: -",
    },
    {
        "Category": "Termination for Convenience",
        "Description": "Can a party terminate this\u00a0 contract without cause (solely by giving a notice and allowing a waiting\u00a0 period to expire)?",
        "Answer Format": "Yes/No",
        "Group": "Group: -",
    },
    {
        "Category": "Rofr/Rofo/Rofn",
        "Description": "Is there a clause granting one party a right of first refusal, right of first offer or right of first negotiation to purchase, license, market, or distribute equity interest, technology, assets, products or services?",
        "Answer Format": "Yes/No",
        "Group": "Group: -",
    },
    {
        "Category": "Change of Control",
        "Description": "Does one party have the right to terminate or is consent or notice required of the counterparty if such party undergoes a change of control, such as a merger, stock sale, transfer of all or substantially all of its assets or business, or assignment by operation of law?",
        "Answer Format": "Yes/No",
        "Group": "Group: 3",
    },
    {
        "Category": "Anti-Assignment",
        "Description": "Is consent or notice required of a party if the contract is assigned to a third party?",
        "Answer Format": "Yes/No",
        "Group": "Group: 3",
    },
    {
        "Category": "Revenue/Profit Sharing",
        "Description": "Is one party required to share revenue or profit with the counterparty for any technology, goods, or\u00a0services?",
        "Answer Format": "Yes/No",
        "Group": "Group: -",
    },
    {
        "Category": "Price Restrictions",
        "Description": "Is there a restriction on the\u00a0 ability of a party to raise or reduce prices of technology, goods, or\u00a0 services provided?",
        "Answer Format": "Yes/No",
        "Group": "Group: -",
    },
    {
        "Category": "Minimum Commitment",
        "Description": "Is there a minimum order size or minimum amount or units per-time period that one party must buy from the counterparty under the contract?",
        "Answer Format": "Yes/No",
        "Group": "Group: -",
    },
    {
        "Category": "Volume Restriction",
        "Description": "Is there a fee increase or consent requirement, etc. if one party\u2019s use of the product/services exceeds certain threshold?",
        "Answer Format": "Yes/No",
        "Group": "Group: -",
    },
    {
        "Category": "IP Ownership Assignment",
        "Description": "Does intellectual property created\u00a0 by one party become the property of the counterparty, either per the terms of the contract or upon the occurrence of certain events?",
        "Answer Format": "Yes/No",
        "Group": "Group: -",
    },
    {
        "Category": "Joint IP Ownership",
        "Description": "Is there any clause providing for joint or shared ownership of intellectual property between the parties to the contract?",
        "Answer Format": "Yes/No",
        "Group": "Group: -",
    },
    {
        "Category": "License Grant",
        "Description": "Does the contract contain a license granted by one party to its counterparty?",
        "Answer Format": "Yes/No",
        "Group": "Group: 4",
    },
    {
        "Category": "Non-Transferable License",
        "Description": "Does the contract limit the ability of a party to transfer the license being granted to a third party?",
        "Answer Format": "Yes/No",
        "Group": "Group: 4",
    },
    {
        "Category": "Affiliate License-Licensor",
        "Description": "Does the contract contain a license grant by affiliates of the licensor or that includes intellectual property of affiliates of the licensor?\u00a0",
        "Answer Format": "Yes/No",
        "Group": "Group: 4",
    },
    {
        "Category": "Affiliate License-Licensee",
        "Description": "Does the contract contain a license grant to a licensee (incl. sublicensor) and the affiliates of such licensee/sublicensor?",
        "Answer Format": "Yes/No",
        "Group": "Group: 4",
    },
    {
        "Category": "Unlimited/All-You-Can-Eat-License",
        "Description": "Is there a clause granting one party an \u201centerprise,\u201d \u201call you can eat\u201d or unlimited usage license?",
        "Answer Format": "Yes/No",
        "Group": "Group: 4",
    },
    {
        "Category": "Irrevocable or Perpetual License",
        "Description": "Does the contract contain a\u00a0 license grant that is irrevocable or perpetual?",
        "Answer Format": "Yes/No",
        "Group": "Group: 4",
    },
    {
        "Category": "Source Code Escrow",
        "Description": "Is one party required to deposit its source code into escrow with a third party, which can be released to the counterparty upon the occurrence of certain events (bankruptcy,\u00a0 insolvency, etc.)?",
        "Answer Format": "Yes/No",
        "Group": "Group: -",
    },
    {
        "Category": "Post-Termination Services",
        "Description": "Is a party subject to obligations after the termination or expiration of a contract, including any post-termination transition, payment, transfer of IP, wind-down, last-buy, or similar commitments?",
        "Answer Format": "Yes/No",
        "Group": "Group: 5",
    },
    {
        "Category": "Audit Rights",
        "Description": "Does a party have the right to\u00a0 audit the books, records, or physical locations of the counterparty to ensure compliance with the contract?",
        "Answer Format": "Yes/No",
        "Group": "Group: 5",
    },
    {
        "Category": "Uncapped Liability",
        "Description": "Is a party\u2019s liability uncapped upon the breach of its obligation in the contract? This also includes uncap liability for a particular type of breach such as IP infringement or breach of confidentiality obligation.",
        "Answer Format": "Yes/No",
        "Group": "Group: 6",
    },
    {
        "Category": "Cap on Liability",
        "Description": "Does the contract include a cap on liability upon the breach of a party\u2019s obligation? This includes time limitation for the counterparty to bring claims or maximum amount for recovery.",
        "Answer Format": "Yes/No",
        "Group": "Group: 6",
    },
    {
        "Category": "Liquidated Damages",
        "Description": "Does the contract contain a clause that would award either party liquidated damages for breach or a fee upon the termination of a contract (termination fee)?",
        "Answer Format": "Yes/No",
        "Group": "Group: -",
    },
    {
        "Category": "Warranty Duration",
        "Description": "What is the duration of any\u00a0 warranty against defects or errors in technology, products, or services\u00a0 provided under the contract?",
        "Answer Format": "Number of months or years",
        "Group": "Group: -",
    },
    {
        "Category": "Insurance",
        "Description": "Is there a requirement for insurance that must be maintained by one party for the benefit of the counterparty?",
        "Answer Format": "Yes/No",
        "Group": "Group: -",
    },
    {
        "Category": "Covenant Not to Sue",
        "Description": "Is a party restricted from contesting the validity of the counterparty\u2019s ownership of intellectual property or otherwise bringing a claim against the counterparty for matters unrelated to the contract?",
        "Answer Format": "Yes/No",
        "Group": "Group: -",
    },
    {
        "Category": "Third Party Beneficiary",
        "Description": "Is there a non-contracting party who is a beneficiary to some or all of the clauses in the contract and therefore can enforce its rights against a contracting party?",
        "Answer Format": "Yes/No",
        "Group": "Group: -",
    },
]

NUM_FIELDS_TO_EXTRACT_PER_CONTRACT = 41

# 0.15 is used in the Doc-ETL paper. It should be 0.5 for the actual benchmark.
IOU_THRESH = 0.15


#  Return the Jaccard similarity between two strings
def get_jaccard(label, pred):
    remove_tokens = [c for c in string.punctuation if c != "/"]
    for token in remove_tokens:
        label = label.replace(token, "")
        pred = pred.replace(token, "")
    label = label.lower()
    pred = pred.lower()
    label = label.replace("/", " ")
    pred = pred.replace("/", " ")

    label_words = set(label.split(" "))
    pred_words = set(pred.split(" "))

    intersection = label_words.intersection(pred_words)
    union = label_words.union(pred_words)
    jaccard = len(intersection) / len(union)
    return jaccard


# Find the number of true positives, false positives, and false negatives for each entry
# (one field extracted from each contract) by comparing the labels and predictions.
# Labels and preds are lists of strings
def evaluate_entry(labels, preds, substr_ok):
    tp, fp, fn = 0, 0, 0

    # jaccard similarity expects strings
    # TODO: This is a hack, ideally, the return type of the preds should be known
    for idx, pred in enumerate(preds):
        if not isinstance(pred, str):
            print(f"Expected string, but got {pred}")
            preds[idx] = str(pred)

    # first check if labels is empty
    if len(labels) == 0:
        if len(preds) > 0:
            fp += len(preds)  # false positive for each one
    else:
        for ans in labels:
            assert len(ans) > 0
            # check if there is a match
            match_found = False
            for pred in preds:
                if substr_ok:
                    is_match = get_jaccard(ans, pred) >= IOU_THRESH or ans in pred
                else:
                    is_match = get_jaccard(ans, pred) >= IOU_THRESH
                if is_match:
                    match_found = True

            if match_found:
                tp += 1
            else:
                fn += 1

        # now also get any fps by looping through preds
        for pred in preds:
            # Check if there's a match. if so, don't count (don't want to double count based on the above)
            # but if there's no match, then this is a false positive.
            # (Note: we get the true positives in the above loop instead of this loop so that we don't double count
            # multiple predictions that are matched with the same answer.)
            match_found = False
            for ans in labels:
                assert len(ans) > 0
                if substr_ok:
                    is_match = get_jaccard(ans, pred) >= IOU_THRESH or ans in pred
                else:
                    is_match = get_jaccard(ans, pred) >= IOU_THRESH
                if is_match:
                    match_found = True

            if not match_found:
                fp += 1

    return tp, fp, fn


# TODO(Siva): This is a temporary fix to handle the case where the preds are empty.
def handle_empty_preds(preds):
    if preds is None or (  # noqa: SIM114
        isinstance(preds, str) and (preds == "" or preds == " " or preds == "null" or preds == "None")
    ):
        return []
    elif isinstance(preds, float) and np.isnan(preds):
        return []
    if not isinstance(preds, (list, np.ndarray)):
        return [preds]
    return preds


# Compute the precision and recall for the entire dataset.
# Each row in the dataframes should correspond to a contract.
# The columns should be the extracted fields (categories in cuad_categories).
def compute_precision_recall(label_df, preds_df):
    tp, fp, fn = 0, 0, 0

    label_df = label_df.sort_values("contract_id").reset_index(drop=True)
    preds_df = preds_df.sort_values("contract_id").reset_index(drop=True)

    assert label_df.shape == preds_df.shape, (
        f"Label and prediction dataframes have different shapes, label shape: {label_df.shape} vs preds shape {preds_df.shape}"
    )

    categories = [category["Category"] for category in cuad_categories]

    for label_row, pred_row in zip(label_df.iterrows(), preds_df.iterrows()):
        assert label_row[1]["contract_id"] == pred_row[1]["contract_id"], (
            f"IDs do not match. label id: {label_row[1]['contract_id']} vs pred id: {pred_row[1]['contract_id']}"
        )
        for category in categories:
            substr_ok = "Parties" in category

            labels = label_row[1][category]
            assert isinstance(labels, list)

            preds = pred_row[1][category]
            preds = handle_empty_preds(preds)

            entry_tp, entry_fp, entry_fn = evaluate_entry(labels, preds, substr_ok)
            tp += entry_tp
            fp += entry_fp
            fn += entry_fn

    precision = tp / (tp + fp) if tp + fp > 0 else np.nan
    recall = tp / (tp + fn) if tp + fn > 0 else np.nan

    return precision, recall

class CUADDataReader(pz.DataReader):
    def __init__(self, num_contracts: int = 1, split: str = "train", seed: int=42):
        self.num_contracts = num_contracts
        self.split = split
        self.seed = seed

        input_cols = [
            {"name": "contract_id", "type": str, "desc": "The id of the the contract to be analyzed"},
            {"name": "title", "type": str, "desc": "The title of the the contract to be analyzed"},
            {"name": "contract", "type": str, "desc": "The content of the the contract to be analyzed"},
        ]
        super().__init__(input_cols)

        # convert the dataset into a list of dictionaries where each row is for a single contract
        include_labels = split == "train"
        dataset = datasets.load_dataset("theatticusproject/cuad-qa")[split]
        self.dataset = self._construct_dataset(dataset, num_contracts, seed, include_labels)


    def _construct_dataset(self, dataset, num_contracts, seed: int=42, include_labels: bool=False):
        # get the set of unique contract titles; to ensure the order of the contracts is
        # preserved, we use a list rather than using python's set()
        contract_titles = []
        for row in dataset:
            if row["title"] not in contract_titles:
                contract_titles.append(row["title"])

        # shuffle the contracts for the given seed
        rng = np.random.default_rng(seed=seed)
        rng.shuffle(contract_titles)

        # get the first num_contracts
        contract_titles = contract_titles[:num_contracts]

        # construct the dataset one contract at a time
        new_dataset = []
        for title in contract_titles:
            # get the rows for this contract
            contract_rows = [row for row in dataset if row["title"] == title]

            # construct the contract; we get the contract_id and contract text from the first row
            contract = {
                "contract_id": contract_rows[0]["id"],
                "title": title,
                "contract": contract_rows[0]["context"],
            }

            # for train / validation data, add the labels
            if include_labels:
                contract = {"fields": contract}

                # add the labels
                category_names = list(map(lambda category: category["Category"], cuad_categories))
                contract["labels"] = {category: [] for category in category_names}
                contract["score_fn"] = {category: None for category in category_names}
                for row in contract_rows:
                    category_name = row["id"].split("__")[-1].split("_")[0].strip()
                    category_name = category_name.replace(" For ", " for ")
                    category_name = category_name.replace(" Of ", " of ")
                    category_name = category_name.replace(" On ", " on ")
                    category_name = category_name.replace(" Or ", " or ")
                    category_name = category_name.replace(" To ", " to ")
                    category_name = category_name.replace("Ip", "IP")
                    assert category_name in category_names, f"Unknown category {category_name}"
                    contract["labels"][category_name].extend(row["answers"]["text"])

                    def score_fn(preds, labels, category_name):
                        preds = handle_empty_preds(preds)
                        entry_tp, _, entry_fn = evaluate_entry(labels, preds, substr_ok=True) if category_name == "Parties" else evaluate_entry(labels, preds, substr_ok=False)
                        score = None
                        if len(labels) > 0:  # noqa: SIM108
                            score = entry_tp / (entry_tp + entry_fn)
                        else:
                            score = 1.0 if len(preds) == 0 else 0.0

                        return score

                    contract["score_fn"][category_name] = partial(score_fn, category_name=category_name)

            # add the rows to the dataset
            new_dataset.append(contract)

        return new_dataset

    def __len__(self):
        return self.num_contracts

    def __getitem__(self, idx: int):
        return self.dataset[idx]

    def get_label_df(self):
        full_dataset = datasets.load_dataset("theatticusproject/cuad-qa")[self.split]
        label_dataset = self._construct_dataset(full_dataset, self.num_contracts, self.seed, True)
        final_label_dataset = []
        for entry in label_dataset:
            row = {}
            row["contract_id"] = entry["fields"]["contract_id"]
            row["title"] = entry["fields"]["title"]
            row["contract"] = entry["fields"]["contract"]
            row = {**row, **entry["labels"]}
            final_label_dataset.append(row)

        return pd.DataFrame(final_label_dataset)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run CUAD demo")
    parser.add_argument("--mode", type=str, help="one-convert or separate-converts", default="one-convert")
    parser.add_argument("--test", type=str, help="test time compute active or inactive", default="active")
    parser.add_argument("--verbose", default=False, action="store_true", help="Print verbose output")
    return parser.parse_args()


def build_cuad_query(dataset, mode):
    assert mode in ["one-convert", "separate-converts"]
    ds = pz.Dataset(dataset)

    if mode == "one-convert":
        cols = []
        for category in cuad_categories:
            desc = (
                f"Extract the text spans (if they exist) from the contract corresponding to {category['Description']}"
            )
            cols.append({"name": category["Category"], "type": ListField(StringField), "desc": desc})

        desc = "Extract the text spans (if they exist) from the contract."
        ds = ds.sem_add_columns(cols, desc=desc, depends_on=["contract"])
    elif mode == "separate-converts":
        for category in cuad_categories:
            desc = (
                f"Extract the text spans (if they exist) from the contract corresponding to {category['Description']}"
            )
            ds = ds.sem_add_columns(
                [{"name": category["Category"], "type": ListField(StringField), "desc": desc}],
                desc=category["Description"],
                depends_on=["contract"],
            )

    return ds


def main():
    if os.getenv("OPENAI_API_KEY") is None and os.getenv("TOGETHER_API_KEY") is None:
        print("WARNING: Both OPENAI_API_KEY and TOGETHER_API_KEY are unset")

    args = parse_arguments()

    # create directory for profiling data
    os.makedirs("opt-profiling-data", exist_ok=True)

    # Create a data reader for the CUAD dataset
    data_reader = CUADDataReader(split="test", num_contracts=1)
    val_data_reader = CUADDataReader(split="train", num_contracts=5)
    print("Created data reader")

    # Build and run the CUAD query
    query = build_cuad_query(data_reader, args.mode)
    print("Built query; Starting query execution")

    processing_strategy = "sentinel"
    execution_strategy = "parallel"
    sentinel_execution_strategy = "all"
    optimizer_strategy = "pareto"
    config = pz.QueryProcessorConfig(
        verbose=False,
        val_datasource=val_data_reader,
        processing_strategy=processing_strategy,
        optimizer_strategy=optimizer_strategy,
        sentinel_execution_strategy=sentinel_execution_strategy,
        execution_strategy=execution_strategy,
        max_workers=64,
        available_models=[
            Model.GPT_4o,
            Model.GPT_4o_MINI,
            # Model.LLAMA3_2_3B,
            Model.LLAMA3_1_8B,
            Model.LLAMA3_3_70B,
            # Model.LLAMA3_2_90B_V,
            Model.MIXTRAL,
            # Model.DEEPSEEK_V3,
            Model.DEEPSEEK_R1_DISTILL_QWEN_1_5B,
        ],
        allow_bonded_query=True,
        allow_code_synth=False,
        allow_critic=True,
        allow_mixtures=True,
        allow_rag_reduction=True,
        progress=True,
    )
    seed = 0
    exp_name = f"cuad-priors-{optimizer_strategy}-seed{seed}"
    data_record_collection = query.run(
        config=config,
        k=-1,
        j=-1,
        sample_budget=1014*5,
        seed=seed,
        exp_name=exp_name,
    )
    print("Query execution completed")

    # save statistics
    execution_stats_dict = data_record_collection.execution_stats.to_json()
    with open(f"priors-data/{exp_name}-stats.json", "w") as f:
        json.dump(execution_stats_dict, f)


if __name__ == "__main__":
    main()
