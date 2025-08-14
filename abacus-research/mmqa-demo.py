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

mmqa_entry_cols = [
    {"name": "qid", "type": str, "desc": "The id of the MMQA question"},
    {"name": "question", "type": str, "desc": "The question which needs to be answered"},
]

mmqa_text_cols = [
    {"name": "supporting_text_ids", "type": list[str], "desc": "A list of text ids for text snippets which may support the question."},
    {"name": "supporting_texts", "type": list[str], "desc": "A list of text snippets which may support the question."},
]

mmqa_table_cols = [
    {"name": "supporting_table_ids", "type": list[str], "desc": "A list of table ids for tables which may support the question."},
    {"name": "supporting_tables", "type": list[str], "desc": "A list of tables which may support the question."},
]

mmqa_image_cols = [
    {"name": "supporting_image_ids", "type": list[str], "desc": "A list of image ids whose images may support the question."},
    {"name": "supporting_images", "type": list[ImageBase64], "desc": "A list of images which may support the question."},
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


class MMQADataset(pz.IterDataset):
    def __init__(
        self,
        num_samples: int = 5,
        split: str = "dev",
        shuffle: bool = False,
        seed: int = 42,
    ):
        super().__init__(id=f"mmqa-{split}", schema=mmqa_entry_cols)

        # read the appropriate dataset
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

        # trim to number of samples
        self.dataset = dataset[:num_samples]
        self.num_samples = num_samples
        self.shuffle = shuffle
        self.seed = seed
        self.split = split

    def compute_label(self, entry: dict) -> dict:
        """Compute the label for a MMQA question given its entry in the dataset."""
        # get the answers
        answers = [answer["answer"] for answer in entry["answers"]]
        supporting_text_doc_ids = [context["doc_id"] for context in entry["supporting_context"] if context["doc_part"] == "text"]
        supporting_table_doc_ids = [context["doc_id"] for context in entry["supporting_context"] if context["doc_part"] == "table"]
        supporting_image_doc_ids = [context["doc_id"] for context in entry["supporting_context"] if context["doc_part"] == "image"]

        # NOTE: inside the optimizer, our qualities will effectively be divided by two,
        #       because we are not providing a label for supporting texts, tables, and images,
        #       however this should be okay b/c it will affect all records equally
        label_dict = {
            "answers": answers,
            "supporting_text_ids": supporting_text_doc_ids,
            "supporting_table_ids": supporting_table_doc_ids,
            "supporting_image_ids": supporting_image_doc_ids,
            "supporting_texts": [],
            "supporting_tables": [],
            "supporting_images": [],
        }

        return label_dict

    @staticmethod
    def recall(preds: list | None, targets: list):
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

    @staticmethod
    def f1(preds: list | None, targets: list):
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

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        # get entry
        entry = self.dataset[idx]

        # get input fields
        qid = entry["qid"]
        question = entry["question"]

        # create item with fields
        item = {"fields": {}, "labels": {}, "score_fn": {}}
        item["fields"]["qid"] = qid
        item["fields"]["question"] = question

        if self.split == "train":
            # add label info
            item["labels"] = self.compute_label(entry)

            # add scoring functions for list fields
            item["score_fn"]["answers"] = MMQADataset.f1
            item["score_fn"]["supporting_text_ids"] = MMQADataset.recall
            item["score_fn"]["supporting_table_ids"] = MMQADataset.recall
            item["score_fn"]["supporting_image_ids"] = MMQADataset.recall

        return item


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
        f"mmqa-final-{sentinel_execution_strategy}-k{k}-j{j}-budget{sample_budget}-seed{seed}"
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

    # create data source
    dataset = MMQADataset(
        split="dev",
        num_samples=100,
        shuffle=True,
        seed=seed,
    )

    # create validation data source
    train_dataset = MMQADataset(
        split="train",
        num_samples=val_examples,
        shuffle=True,
        seed=seed,
    )

    # load index [text-embedding-3-small]
    chroma_client = chromadb.PersistentClient(".chroma-mmqa")
    openai_ef = OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"],
        model_name="text-embedding-3-small",
    )
    sentence_transformer_ef = SentenceTransformerEmbeddingFunction(
        model_name="clip-ViT-B-32"
    )
    text_index = chroma_client.get_collection("mmqa-texts", embedding_function=openai_ef)
    table_index = chroma_client.get_collection("mmqa-tables", embedding_function=openai_ef)
    image_index = chroma_client.get_collection("mmqa-images", embedding_function=sentence_transformer_ef)

    def get_results_and_ids(index: chromadb.Collection, query: list[list[float]], n_results: int, image=False) -> tuple[list[str]]:
        # execute query with embeddings
        results = index.query(query, n_results=n_results)

        # get list of result terms with their cosine similarity scores
        final_results = []
        for query_doc_ids, query_docs, query_distances in zip(results["ids"], results["documents"], results["distances"]):
            for doc_id, doc, dist in zip(query_doc_ids, query_docs, query_distances):
                cosine_similarity = 1 - dist
                final_results.append({"content": doc, "id": doc_id, "similarity": cosine_similarity})

        # sort the results by similarity score
        sorted_results = sorted(final_results, key=lambda result: result["similarity"], reverse=True)

        # remove duplicates
        sorted_results_set = set()
        final_sorted_results, final_sorted_result_ids = [], []
        for result in sorted_results:
            if result["content"] not in sorted_results_set:
                sorted_results_set.add(result["content"])
                final_sorted_results.append(result["content"])
                final_sorted_result_ids.append(result["id"])

        # return the top-k similar results and generation stats
        return final_sorted_results[:n_results], final_sorted_result_ids[:n_results]

    def text_search_func(index: chromadb.Collection, query: list[list[float]], k: int) -> list[str]:
        # execute query with embeddings
        results, result_ids = get_results_and_ids(index, query, n_results=k)
        return {"supporting_texts": results, "supporting_text_ids": result_ids}

    def table_search_func(index: chromadb.Collection, query: list[list[float]], k: int) -> list[str]:
        # execute query with embeddings
        results, result_ids = get_results_and_ids(index, query, n_results=k)
        return {"supporting_tables": results, "supporting_table_ids": result_ids}

    def image_search_func(index: chromadb.Collection, query: list[list[float]], k: int) -> list[str]:
        # limit max number of results to 5
        # k = min(k, 5)

        # execute query with embeddings
        _, result_ids = get_results_and_ids(index, query, n_results=k, image=True)
        possible_endings = {'.JPG', '.png', '.jpeg', '.jpg', '.tif', '.JPEG', '.tiff', '.PNG', '.Jpg', '.gif'}

        results = []
        for image_id in result_ids:
            # find the correct image file
            for ending in possible_endings:
                if os.path.exists(f"testdata/mmqa-images/{image_id}{ending}"):
                    image_id += ending
                    break

            # load image from disk
            with open(f"testdata/mmqa-images/{image_id}", "rb") as f:
                base64_image_str = base64.b64encode(f.read())
                results.append(base64_image_str)

        return {"supporting_images": results, "supporting_image_ids": result_ids}

    # construct plan
    plan = dataset.retrieve(
        index=text_index,
        search_func=text_search_func,
        search_attr="question",
        output_attrs=mmqa_text_cols,
    )
    plan = plan.retrieve(
        index=table_index,
        search_func=table_search_func,
        search_attr="question",
        output_attrs=mmqa_table_cols,
    )
    plan = plan.retrieve(
        index=image_index,
        search_func=image_search_func,
        search_attr="question",
        output_attrs=mmqa_image_cols,
    )
    plan = plan.sem_add_columns(mmqa_answer_cols)

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

    data_record_collection = plan.run(config=config, train_dataset=train_dataset, validator=pz.Validator())

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

    # # save statistics
    # execution_stats_dict = data_record_collection.execution_stats.to_json()
    # with open(stats_path, "w") as f:
    #     json.dump(execution_stats_dict, f)

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
