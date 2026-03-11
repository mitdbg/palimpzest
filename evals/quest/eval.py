import argparse
import copy
import json
import os
import random
import time

import palimpzest as pz


def prepare_docs_for_query(items: list, gt_docs: list) -> list:
    items = copy.copy(items)
    random.shuffle(items)
    final_items = [doc for doc in items if doc["title"] in gt_docs]
    while len(final_items) < 1000 and len(items) > 0:
        item = items.pop(0)
        if item not in final_items:
            final_items.append(item)
    return final_items


def palimpzest_run_query(query: dict, documents: list) -> list[str]:
    gt_docs = query["docs"]
    items = prepare_docs_for_query(documents, gt_docs)

    schema = [
        {"name": "title", "type": str, "desc": "Document title"},
        {"name": "text", "type": str, "desc": "Document content"},
    ]

    dataset = pz.MemoryDataset(
        id="quest-docs",
        vals=items,
        schema=schema,
    )

    query_text = query["query"]
    plan = dataset.sem_filter(
        f'This document is relevant to the entity-seeking query: "{query_text}". '
        "Return True if the document helps answer the query, False otherwise.",
        depends_on=["text"],
    ).project(["title"])

    config = pz.QueryProcessorConfig(
        policy=pz.MaxQuality(),
        execution_strategy="parallel",
        progress=True,
    )
    output = plan.run(config)
    execution_stats = output.execution_stats
    time_secs = execution_stats.total_execution_time if execution_stats else 0.0
    cost = execution_stats.total_execution_cost if execution_stats else 0.0
    return [record["title"] for record in output], time_secs, cost


def main():
    parser = argparse.ArgumentParser(description="Evaluate Palimpzest on QUEST")
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        choices=["films", "books"],
        help="The domain to evaluate.",
    )
    parser.add_argument(
        "--queries",
        type=str,
        required=True,
        help="Path to the file containing the queries (e.g. test.jsonl).",
    )
    parser.add_argument(
        "--documents",
        type=str,
        default="data/documents.jsonl",
        help="Path to documents.jsonl (QUEST format: title, text per line).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of queries to evaluate (for debugging).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for document shuffling.",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    if not os.path.exists(args.documents):
        raise FileNotFoundError(
            f"Documents file not found: {args.documents}\n"
        )
    with open(args.documents) as f:
        documents = [json.loads(line) for line in f]

    queries = []
    with open(args.queries) as f:
        for line in f:
            d = json.loads(line)
            if d["metadata"]["domain"] == args.domain:
                queries.append(d)

    if args.limit:
        queries = queries[: args.limit]

    results = []
    for i, query in enumerate(queries):
        print(f"[{i + 1}/{len(queries)}] Executing query: {query['query']}")
        pred_docs, cur_time, cur_cost = palimpzest_run_query(query, documents)

        gt_docs = query["docs"]
        preds = set(pred_docs)
        labels = set(gt_docs)

        tp = sum(1 for pred in preds if pred in labels)
        fp = len(preds) - tp
        fn = sum(1 for label in labels if label not in preds)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        result = {
            "query": query["query"],
            "predicted_docs": pred_docs,
            "ground_truth_docs": gt_docs,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "time": cur_time,
            "cost": cur_cost
        }
        results.append(result)

    ts = int(time.time())
    out_path = f"results_{args.domain}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {out_path}")

    n = len(results)
    avg_precision = sum(r["precision"] for r in results) / n
    avg_recall = sum(r["recall"] for r in results) / n
    avg_f1 = sum(r["f1_score"] for r in results) / n
    avg_time = sum(r["time"] for r in results) / n
    avg_cost = sum(r["cost"] for r in results) / n

    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")
    print(f"Average Time: {avg_time:.4f}s")
    print(f"Average Cost: {avg_cost:.4f}$")

if __name__ == "__main__":
    main()
