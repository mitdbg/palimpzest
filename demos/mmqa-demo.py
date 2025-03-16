import argparse
import base64
import json
import os
import random
import time

import chromadb
from chromadb.utils.embedding_functions.openai_embedding_function import OpenAIEmbeddingFunction

import palimpzest as pz
from palimpzest.constants import Model
from palimpzest.core.lib.fields import ImageBase64Field, ListField
from palimpzest.utils.model_helpers import get_models

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
    {"name": "supporting_images", "type": ListField(ImageBase64Field), "desc": "A list of images which may support the question."},
]

mmqa_answer_cols = [
    {"name": "answers", "type": list[str], "desc": "The answer(s) to the question. This is a list of strings which will be a singleton if there is only one answer."},
]


class MMQAReader(pz.DataReader):
    def __init__(
        self,
        num_samples: int = 5,
        split: str = "test",
        shuffle: bool = False,
        seed: int = 42,
    ):
        super().__init__(mmqa_entry_cols)

        # read the appropriate dataset
        dataset = []
        with open(f"testdata/MMQA_{split}.jsonl") as f:
            for line in f:
                dict_line = json.loads(line)
                if split == "train" and "image" in dict_line["metadata"]["modalities"]:  # noqa: SIM114
                    dataset.append(dict_line)
                elif split == "test" and len(dict_line["metadata"]["image_doc_ids"]) > 0:
                    dataset.append(dict_line)

        # shuffle records if shuffle = True
        if shuffle:
            random.Random(seed).shuffle(dataset)

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

        try:
            # compute recall of retrieved ids and return
            intersect = set(preds).intersection(set(targets))
            recall = len(intersect) / len(targets)

            return recall

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
            item["score_fn"]["answers"] = MMQAReader.recall
            item["score_fn"]["supporting_text_ids"] = MMQAReader.recall
            item["score_fn"]["supporting_table_ids"] = MMQAReader.recall
            item["score_fn"]["supporting_image_ids"] = MMQAReader.recall

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
    datareader = MMQAReader(
        split="test",
        num_samples=100,
        shuffle=False,
        seed=seed,
    )

    # create validation data source
    val_datasource = MMQAReader(
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
    text_index = chroma_client.get_collection("mmqa-texts", embedding_function=openai_ef)
    table_index = chroma_client.get_collection("mmqa-tables", embedding_function=openai_ef)
    image_index = chroma_client.get_collection("mmqa-image-titles", embedding_function=openai_ef)

    def get_results_and_ids(index: chromadb.Collection, query: list[list[float]], n_results: int) -> tuple[list[str]]:
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
        return final_sorted_results[:k], final_sorted_result_ids[:k]

    def text_search_func(index: chromadb.Collection, query: list[list[float]], k: int) -> list[str]:
        # execute query with embeddings
        results, result_ids = get_results_and_ids(index, query, n_results=k)
        return {"supporting_texts": results, "supporting_text_ids": result_ids}

    def table_search_func(index: chromadb.Collection, query: list[list[float]], k: int) -> list[str]:
        # execute query with embeddings
        results, result_ids = get_results_and_ids(index, query, n_results=k)
        return {"supporting_tables": results, "supporting_table_ids": result_ids}

    def image_search_func(index: chromadb.Collection, query: list[list[float]], k: int) -> list[str]:
        # execute query with embeddings
        _, result_ids = get_results_and_ids(index, query, n_results=k)
        possible_endings = {'JPG', 'png', 'jpeg', 'jpg', 'tif', 'JPEG', 'tiff', 'PNG', 'Jpg', 'gif'}

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
    plan = pz.Dataset(datareader)
    plan = plan.retrieve(
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
        max_workers=1,
        verbose=verbose,
        available_models=[
            # Model.GPT_4o,
            # Model.GPT_4o_V,
            Model.GPT_4o_MINI,
            Model.GPT_4o_MINI_V,
            # Model.DEEPSEEK,
            # Model.MIXTRAL,
            # Model.LLAMA3,
            # Model.LLAMA3_V,
        ],
        allow_bonded_query=True,
        allow_code_synth=False,
        allow_critic=False,
        allow_mixtures=False,
        allow_rag_reduction=False,
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
    data_record_collection.to_df().to_csv(f"opt-profiling-data/mmqa-{exp_name}-output.csv", index=False)

    # create filepaths for records and stats
    records_path = f"opt-profiling-data/mmqa-{exp_name}-records.json"
    stats_path = f"opt-profiling-data/mmqa-{exp_name}-profiling.json"

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

    # save statistics
    execution_stats_dict = data_record_collection.execution_stats.to_json()
    with open(stats_path, "w") as f:
        json.dump(execution_stats_dict, f)
