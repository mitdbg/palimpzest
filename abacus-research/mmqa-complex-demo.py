import argparse
import base64
import json
import os
import string
import time

import chromadb
import numpy as np
import regex as re
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)
from chromadb.utils.embedding_functions.openai_embedding_function import (
    OpenAIEmbeddingFunction,
)

import palimpzest as pz
from palimpzest.constants import Model
from palimpzest.core.lib.schemas import ImageBase64

CORRUPTED_IMAGE_IDS = [
    "17ae0616ac745e70781203267f3a382d",
    "bf201cbbd058ef51aef89b1be4158c2a",
    "ef457a7b3ab437cd78ab9f82dc083048",
    "225c3db49d60b5ef30ed0bfc649ebf78",
    "b413cc1dc4969dcbe4cb6a55c0f2e359",
    "e81b2acfd792b171389c8f47a0e14504",
]

mmqa_entry_cols = [
    {"name": "qid", "type": str, "desc": "The id of the MMQA question"},
    {"name": "question", "type": str, "desc": "The question which needs to be answered"},
]
mmqa_text_search_cols = [
    {"name": "text_search_string", "type": str, "desc": "A string used to search for relevant text snippets."},
]
mmqa_table_search_cols = [
    {"name": "table_search_string", "type": str, "desc": "A string used to search for relevant tables."},
]
mmqa_image_search_cols = [
    {"name": "image_search_string", "type": str, "desc": "A string used to search for relevant images."},
]

mmqa_text_cols = [
    {"name": "text_id", "type": str, "desc": "The id for the given text snippet."},
    {"name": "text", "type": str, "desc": "A text snippet which may or may not support the question."},
]

mmqa_table_cols = [
    {"name": "table_id", "type": str, "desc": "The id for the given table."},
    {"name": "table", "type": str, "desc": "A table which may or may not support the question."},
]

mmqa_image_cols = [
    {"name": "image_id", "type": str, "desc": "The id for the given image."},
    {"name": "image", "type": ImageBase64, "desc": "An image which may or may not support the question."},
]

mmqa_answer_cols = [
    {"name": "answers", "type": list[str], "desc": "The answer(s) to the question. Answer the question using the relevant information from gathered image(s), text(s), and table(s). Return your answer as a JSON list of strings. Do not include any additional context or an explanation in your answer, simply list the entities asked for by the question"},
]

def get_json_from_answer(answer: str):
    """
    This function parses an LLM response which is supposed to output a JSON object
    and optimistically searches for the substring containing the JSON object.
    """
    # split off context / excess, which models sometimes output after answer
    answer = answer.split("Context:")[0]
    answer = answer.split("# this is the answer")[0]
    # trim the answer to only include the JSON array
    if not answer.strip().startswith("["):
        # Find the start index of the actual JSON string assuming the prefix is followed by the JSON array
        start_index = answer.find("[")
        if start_index != -1:
            # Remove the prefix and any leading characters before the JSON starts
            answer = answer[start_index:]
    if not answer.strip().endswith("]"):
        # Find the end index of the actual JSON string
        # assuming the suffix is preceded by the JSON object/array
        end_index = answer.rfind("]")
        if end_index != -1:
            # Remove the suffix and any trailing characters after the JSON ends
            answer = answer[: end_index + 1]
    # Handle weird escaped values. I am not sure why the model
    # is returning these, but the JSON parser can't take them
    answer = answer.replace(r"\_", "_")
    answer = answer.replace("\\n", "\n")
    # Remove https and http prefixes to not conflict with comment detection
    # Handle comments in the JSON response. Use regex from // until end of line
    answer = re.sub(r"(?<!https?:)\/\/.*?$", "", answer, flags=re.MULTILINE)
    answer = re.sub(r",\n.*\.\.\.$", "", answer, flags=re.MULTILINE)
    # Sanitize newlines in the JSON response
    answer = answer.replace("\n", " ")
    # finally, parse and return the JSON object; errors are handled by the caller
    return json.loads(answer)


class MMQAValidator(pz.Validator):
    def __init__(self, dataset: list[dict]):
        super().__init__()
        self.dataset = dataset

        # compute qid to label mapping
        self.qid_to_labels = self._compute_qid_to_labels()

    def _compute_qid_to_labels(self) -> dict:
        """Compute the label for a MMQA question given its entry in the dataset."""
        qid_to_labels = {}
        for entry in self.dataset:
            # get the answers
            answers = [answer["answer"] for answer in entry["answers"]]
            supporting_text_ids = [context["doc_id"] for context in entry["supporting_context"] if context["doc_part"] == "text"]
            supporting_table_ids = [context["doc_id"] for context in entry["supporting_context"] if context["doc_part"] == "table"]
            supporting_image_ids = [context["doc_id"] for context in entry["supporting_context"] if context["doc_part"] == "image"]

            label_dict = {
                "answers": answers,
                "supporting_text_ids": supporting_text_ids,
                "supporting_table_ids": supporting_table_ids,
                "supporting_image_ids": supporting_image_ids,
            }
            qid_to_labels[entry["qid"]] = label_dict

        return qid_to_labels

    def recall(self, preds: list | None, targets: list):
        if preds is None or len(targets) == 0:
            return 0.0

        tp, fn = 0, 0
        try:
            # compute recall of retrieved ids and return
            preds = [str(pred).lower() for pred in preds]
            targets = [str(target).lower() for target in targets]
            remove_tokens = [c for c in string.punctuation if c != "/"]
            for token in remove_tokens:
                preds = [pred.replace(token, "") for pred in preds]
                targets = [target.replace(token, "") for target in targets]


            for target in targets:
                if target in preds:
                    tp += 1
                else:
                    fn += 1

            return tp / (tp + fn)

        except Exception:
            os.makedirs("mmqa-recall-eval-errors", exist_ok=True)
            ts = time.time()
            with open(f"mmqa-recall-eval-errors/error-{ts}.txt", "w") as f:
                f.write(str(preds))
            return 0.0

    def f1(self, preds: list | None, targets: list):
        if preds is None or len(targets) == 0:
            return 0.0

        tp, fp, fn = 0, 0, 0
        try:
            # compute recall of retrieved ids and return
            preds = [str(pred).lower() for pred in preds]
            targets = [str(target).lower() for target in targets]

            remove_tokens = [c for c in string.punctuation if c != "/"]
            for token in remove_tokens:
                preds = [pred.replace(token, "") for pred in preds]
                targets = [target.replace(token, "") for target in targets]

            for pred in preds:
                if pred in targets:
                    tp += 1
                else:
                    fp += 1
            for target in targets:
                if target not in preds:
                    fn += 1

            # compute overall f1 score and return
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            return f1

        except Exception:
            os.makedirs("mmqa-recall-eval-errors", exist_ok=True)
            ts = time.time()
            with open(f"mmqa-recall-eval-errors/error-{ts}.txt", "w") as f:
                f.write(str(preds))
            return 0.0

    def map_score_fn(self, fields: list[str], input_record: dict, output: dict) -> float | None:
        if "answers" not in fields:
            return None
        preds = output.get("answers")
        targets = self.qid_to_labels[str(input_record["qid"])]["answers"]
        return self.f1(preds, targets)

    def join_score_fn(self, condition: str, left_input_record: dict, right_input_record: dict, output: bool) -> float | None:
        if condition == "The text snippet is relevant to the question based on the text search string.":
            pred = right_input_record["text_id"]
            targets = self.qid_to_labels[left_input_record["qid"]]["supporting_text_ids"]
            return pred in targets and output or pred not in targets and not output
        elif condition == "The table is relevant to the question based on the table search string.":
            pred = right_input_record["table_id"]
            targets = self.qid_to_labels[left_input_record["qid"]]["supporting_table_ids"]
            return pred in targets and output or pred not in targets and not output
        elif condition == "The image is relevant to the question based on the image search string.":
            pred = right_input_record["image_id"]
            targets = self.qid_to_labels[left_input_record["qid"]]["supporting_image_ids"]
            return pred in targets and output or pred not in targets and not output
        else:
            raise NotImplementedError(f"Validator.retrieve_score_fn not implemented for condition {condition}.")


class MMQAQuestionDataset(pz.IterDataset):
    def __init__(self, dataset: list[dict]):
        super().__init__(id="mmqa-questions", schema=mmqa_entry_cols)
        self.dataset = [{"qid": entry["qid"], "question": entry["question"]} for entry in dataset]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        return self.dataset[idx]


class MMQATextDataset(pz.IterDataset):
    def __init__(self, dataset: list[dict]):
        super().__init__(id="mmqa-texts", schema=mmqa_entry_cols)

        # construct mapping from text id to text
        text_id_to_text = {}
        with open("data/MMQA_texts.jsonl") as f:
            for line in f:
                dict_line = json.loads(line)
                text_id_to_text[dict_line["id"]] = f"{dict_line['title']}: {dict_line['text']}"

        # construct dataset
        self.dataset = []
        for entry in dataset:
            for context in entry["supporting_context"]:
                if context["doc_part"] == "text":
                    text_id = context["doc_id"]
                    text = text_id_to_text[text_id]
                    self.dataset.append({"text_id": text_id, "text": text})

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        return self.dataset[idx]


class MMQATableDataset(pz.IterDataset):
    def __init__(self, dataset: list[dict]):
        super().__init__(id="mmqa-tables", schema=mmqa_entry_cols)

        # construct mapping from table id to table string
        table_id_to_table = {}
        with open("data/MMQA_tables.jsonl") as f:
            for line in f:
                dict_line = json.loads(line)

                # get page title and table name
                page_title = dict_line["title"]
                table_name = dict_line["table"]["table_name"]

                # get table column names and empty column indices
                table_header = dict_line["table"]["header"]
                column_names = [col["column_name"] for col in table_header if col["column_name"] != ""]
                empty_col_indices = set([idx for idx, col in enumerate(table_header) if col["column_name"] == ""])

                # create string for table data
                text = f"{page_title}: {table_name}\n\n{','.join(column_names)}\n"

                # parse table row data
                table_rows = dict_line["table"]["table_rows"]
                for row in table_rows:
                    row_data = []
                    for idx, cell in enumerate(row):
                        if idx in empty_col_indices:
                            continue
                        row_data.append(cell["text"])

                    text += ",".join(row_data) + "\n"

                table_id_to_table[dict_line["id"]] = text

        # construct dataset
        self.dataset = []
        for entry in dataset:
            for context in entry["supporting_context"]:
                if context["doc_part"] == "table":
                    table_id = context["doc_id"]
                    table = table_id_to_table[table_id]
                    self.dataset.append({"table_id": table_id, "table": table})

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        return self.dataset[idx]


class MMQAImageDataset(pz.IterDataset):
    def __init__(self, dataset: list[dict]):
        super().__init__(id="mmqa-images", schema=mmqa_entry_cols)

        # construct mapping from image id to image base64 object
        image_id_to_image = {}
        with open("data/MMQA_images.jsonl") as f:
            possible_endings = {'.JPG', '.png', '.jpeg', '.jpg', '.tif', '.JPEG', '.tiff', '.PNG', '.Jpg', '.gif'}
            for line in f:
                dict_line = json.loads(line)
                image_id = dict_line["id"]

                # skip corrupted images:
                if image_id in CORRUPTED_IMAGE_IDS:
                    continue

                # find the correct image file
                image_filepath = None
                for ending in possible_endings:
                    filepath = f"data/final_dataset_images/{image_id}{ending}"
                    if os.path.exists(filepath):
                        image_filepath = filepath
                        break

                # read the image file and convert to base64
                with open(image_filepath, "rb") as f:
                    contents = base64.b64encode(f.read()).decode("utf-8")
                    image_id_to_image[image_id] = contents

        # construct dataset
        self.dataset = []
        for entry in dataset:
            for context in entry["supporting_context"]:
                if context["doc_part"] == "image":
                    image_id = context["doc_id"]
                    image = image_id_to_image[image_id]
                    self.dataset.append({"image_id": image_id, "image": image})

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        return self.dataset[idx]


def get_dataset(split: str, shuffle: bool, seed: int, num_samples: int | None) -> list[str]:
    dataset = []
    with open(f"data/MMQA_{split}.jsonl") as f:
        for line in f:
            dict_line = json.loads(line)
            if "image" in dict_line["metadata"]["modalities"] and len(dict_line["metadata"]["modalities"]) > 1:
                dataset.append(dict_line)

    # shuffle the questions for the given seed
    if shuffle:
        rng = np.random.default_rng(seed=seed)
        rng.shuffle(dataset)
    
    return dataset if num_samples is None else dataset[:num_samples]


def compute_f1(final_df, answers_df):
    merged_df = final_df.merge(answers_df, on="qid", how="left")
    tp, fp, fn = 0, 0, 0
    for _, row in merged_df.iterrows():
        targets = [str(target).lower() for target in row["gt_answers"]]
        preds = row["answers"]
        if isinstance(preds, str):
            try:
                # convert single quotes to double quotes before parsing for JSON
                preds = preds.replace("'", '"')
                # try parsing preds as JSON list and cast everything to str to match targets
                preds = get_json_from_answer(preds)
                preds = [str(pred).lower() for pred in preds]
            except Exception:
                # if that fails, give it a shot as a singleton answer that the LLM failed to wrap in a list
                preds = [preds.lower()]
            remove_tokens = [c for c in string.punctuation if c != "/"]
            for token in remove_tokens:
                preds = [pred.replace(token, "") for pred in preds]
                targets = [target.replace(token, "") for target in targets]
        elif isinstance(preds, list):
            preds = [str(pred).lower() for pred in preds]
            remove_tokens = [c for c in string.punctuation if c != "/"]
            for token in remove_tokens:
                preds = [pred.replace(token, "") for pred in preds]
                targets = [target.replace(token, "") for target in targets]
        else:
            preds = []
        for pred in preds:
            if pred in targets:
                tp += 1
            else:
                fp += 1
        for target in targets:
            if target not in preds:
                fn += 1
    # compute overall f1 score and return
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description="Run a simple demo")
    parser.add_argument("--verbose", default=False, action="store_true", help="Print verbose output")
    parser.add_argument("--progress", default=False, action="store_true", help="Print progress output")
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
        default=20,
        type=int,
        help="Number of validation examples to sample from",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        type=str,
        help="One of 'gpt-4o', 'gpt-4o-mini', 'llama'",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Seed used to initialize RNG for MAB sampling algorithm",
    )
    parser.add_argument(
        "--k",
        default=6,
        type=int,
        help="Number of columns to sample in Random Sampling or MAB sentinel execution",
    )
    parser.add_argument(
        "--j",
        default=4,
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
        "--quality",
        default=None,
        type=float,
        help="Quality threshold",
    )
    parser.add_argument(
        "--exp-name",
        default=None,
        type=str,
        help="The experiment name.",
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
        f"mmqa-complex-final-{sentinel_execution_strategy}-k{k}-j{j}-budget{sample_budget}-seed{seed}"
        if args.exp_name is None
        else args.exp_name
    )

    policy = pz.MaxQuality()
    if args.quality is not None and args.policy == "mincostatfixedquality":
        policy = pz.MinCostAtFixedQuality(min_quality=args.quality)
    elif args.quality is not None and args.policy == "minlatencyatfixedquality":
        policy = pz.MinTimeAtFixedQuality(min_quality=args.quality)
    print(f"USING POLICY: {policy}")

    if os.getenv("OPENAI_API_KEY") is None and os.getenv("TOGETHER_API_KEY") is None and os.getenv("ANTHROPIC_API_KEY") is None:
        print("WARNING: OPENAI_API_KEY, TOGETHER_API_KEY, and ANTHROPIC_API_KEY are unset")

    # create the train and test dataset
    train_dataset = get_dataset(split="train", shuffle=True, seed=seed, num_samples=val_examples)
    test_dataset = get_dataset(split="dev", shuffle=True, seed=seed, num_samples=100)

    # create validator for MMQA
    validator = MMQAValidator(train_dataset)

    # create train datasets for questions, texts, tables, and images
    train_question_dataset = MMQAQuestionDataset(train_dataset)
    train_text_dataset = MMQATextDataset(train_dataset)
    train_table_dataset = MMQATableDataset(train_dataset)
    train_image_dataset = MMQAImageDataset(train_dataset)
    train_dataset = {
        train_question_dataset.id: train_question_dataset,
        train_text_dataset.id: train_text_dataset,
        train_table_dataset.id: train_table_dataset,
        train_image_dataset.id: train_image_dataset,
    }

    # construct plan
    test_question_dataset = MMQAQuestionDataset(test_dataset)
    test_text_dataset = MMQATextDataset(test_dataset)
    test_table_dataset = MMQATableDataset(test_dataset)
    test_image_dataset = MMQAImageDataset(test_dataset)
    text_plan = test_question_dataset.sem_map(mmqa_text_search_cols, depends_on=["question"])
    text_plan = text_plan.sem_join(
        test_text_dataset,
        condition="The text snippet is relevant to the question based on the text search string.",
        depends_on=["text_search_string", "text"],
        how="left",
    )
    text_plan = text_plan.groupby(pz.GroupBySig(["qid", "question", "text_search_string"], agg_funcs=["list", "list"], agg_fields=["text_id", "text"]))
    text_plan = text_plan.map(
        udf=lambda record: "...".join(record["list(text)"]) if record["list(text)"] != [None] else "None",
        cols=[{"name": "text", "type": str, "desc": "All relevant text snippets concatenated together."}],
    )
    text_plan = text_plan.project(["qid", "question", "text"])

    table_plan = test_question_dataset.sem_map(mmqa_table_search_cols, depends_on=["question"])
    table_plan = table_plan.sem_join(
        test_table_dataset,
        condition="The table is relevant to the question based on the table search string.",
        depends_on=["table_search_string", "table"],
        how="left",
    )
    table_plan = table_plan.groupby(pz.GroupBySig(["qid", "question", "table_search_string"], agg_funcs=["list", "list"], agg_fields=["table_id", "table"]))
    table_plan = table_plan.map(
        udf=lambda record: "...".join(record["list(table)"]) if record["list(table)"] != [None] else "None",
        cols=[{"name": "table", "type": str, "desc": "All relevant table snippets concatenated together."}],
    )
    table_plan = table_plan.project(["qid", "question", "table"])

    image_plan = test_question_dataset.sem_map(mmqa_image_search_cols, depends_on=["question"])
    image_plan = image_plan.sem_join(
        test_image_dataset,
        condition="The image is relevant to the question based on the image search string.",
        depends_on=["image_search_string", "image"],
    )
    image_plan = image_plan.groupby(pz.GroupBySig(["qid", "question", "image_search_string"], agg_funcs=["list", "list"], agg_fields=["image_id", "image"]))
    image_plan = image_plan.map(
        udf=lambda record: "...".join(record["list(image)"]) if record["list(image)"] != [None] else "None",
        cols=[{"name": "image", "type": str, "desc": "All relevant image snippets concatenated together."}],
    )
    image_plan = image_plan.project(["qid", "question", "image"])

    plan = text_plan.join(table_plan, on=["qid", "question"]).join(image_plan, on=["qid", "question"])
    plan = plan.sem_map(mmqa_answer_cols, depends_on=["question", "text", "table", "image"])

    # TODO:
    # 1. add "how" argument to sem_join
    #    a. have left / right outer join return None for missing values
    # 2. add join operator

    # execute pz plan
    config = pz.QueryProcessorConfig(
        policy=policy,
        optimizer_strategy="pareto",
        sentinel_execution_strategy=sentinel_execution_strategy,
        execution_strategy=execution_strategy,
        use_final_op_quality=True,
        max_workers=1,
        verbose=verbose,
        available_models=[
            Model.GPT_4o_MINI,
        ],
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
    )

    data_record_collection = plan.optimize_and_run(config=config, train_dataset=train_dataset, validator=validator)

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
            if k in ["qid", "question", "supporting_text_ids", "supporting_table_ids", "supporting_image_ids", "answers"]
        }
        record_jsons.append(record_dict)

    with open(records_path, "w") as f:
        json.dump(record_jsons, f)

    # read the appropriate dataset
    dataset = []
    with open("data/MMQA_dev.jsonl") as f:
        for line in f:
            dict_line = json.loads(line)
            if "image" in dict_line["metadata"]["modalities"] and len(dict_line["metadata"]["modalities"]) > 1:
                dataset.append(dict_line)

    # shuffle the questions for the given seed
    rng = np.random.default_rng(seed=seed)
    rng.shuffle(dataset)

    # trim to 100 samples
    dataset = dataset[:100]
    answer_dataset = []
    for item in dataset:
        answers = list(map(lambda elt: str(elt["answer"]), item["answers"]))
        answer_dataset.append({
            "qid": item["qid"],
            "gt_answers": answers
        })

    # construction dataframe
    import pandas as pd
    answers_df = pd.DataFrame(answer_dataset)

    # get final plan str
    final_plan_id = list(data_record_collection.execution_stats.plan_stats.keys())[0]
    final_plan_str = data_record_collection.execution_stats.plan_strs[final_plan_id]

    # write stats to disk
    stats_dict = {
        "f1": compute_f1(data_record_collection.to_df(), answers_df),
        "optimization_time": data_record_collection.execution_stats.optimization_time,
        "optimization_cost": data_record_collection.execution_stats.optimization_cost,
        "plan_execution_time": data_record_collection.execution_stats.plan_execution_time,
        "plan_execution_cost": data_record_collection.execution_stats.plan_execution_cost,
        "total_execution_time": data_record_collection.execution_stats.total_execution_time,
        "total_execution_cost": data_record_collection.execution_stats.total_execution_cost,
        "plan_str": final_plan_str,
    }
    print(f"F1 IS: {stats_dict['f1']}")

    with open(f"opt-profiling-data/{exp_name}-stats.json", "w") as f:
        json.dump(stats_dict, f)
