#!/usr/bin/env python3
"""
Query 2 — Critic Volume by Inferred Era (Palimpzest)

Group reviews by movie era (pre-2000, 2000s, 2010s, 2020s) and count reviews.
The LLM semantically infers the era from the releaseDateTheaters column.

Pipeline:
  1. Join movie_reviews with movies to get releaseDateTheaters.
  2. sem_groupby – LLM reads releaseDateTheaters and groups into era buckets;
                   counts reviewId per group.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

repo_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(repo_root / "src"))

import palimpzest as pz

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Reviews can be categorized into pre-2000, 2000s, 2010s, 2020s, or unknown. Return which era category the review falls into")
    parser.add_argument("--verbose", default=False, action="store_true")
    parser.add_argument("--policy", type=str, default="maxquality")
    parser.add_argument("--output", type=str, default="query2_pz_output.csv")
    parser.add_argument("--stats-output", type=str, default=None,
                        help="Optional path to write execution stats JSON")
    parser.add_argument(
        "--execution-strategy", type=str, default="sequential",
        help="One of 'sequential', 'pipelined', 'parallel'",
    )
    args = parser.parse_args()

    policy_map = {
        "mincost": pz.MinCost(),
        "mintime": pz.MinTime(),
        "maxquality": pz.MaxQuality(),
    }
    policy = policy_map.get(args.policy, pz.MaxQuality())

    script_dir = Path(__file__).parent

    # Load and join data
    reviews_df = pd.read_csv(script_dir / "../movie_reviews.csv").head(500)
    movies_df = pd.read_csv(script_dir / "../movies.csv")[["id", "releaseDateTheaters"]]
    merged_df = reviews_df.merge(movies_df, on="id", how="left")
    print(f"Loaded {len(merged_df)} reviews")

    reviews = pz.MemoryDataset(id="reviews", vals=merged_df)

    # sem_groupby: LLM infers era from releaseDateTheaters, count reviewId per era
    grouped = reviews.sem_groupby(
        gby_fields=[
            {
                "name": "releaseDateTheaters",
                "type": str,
                "desc": "Reviews can be categorized into pre-2000, 2000s, 2010s, 2020s, or unknown. Return which era category the review falls into)",
            }
        ],
        agg_fields=[
            {
                "name": "reviewId",
                "type": int,
                "desc": "Identifier of the review",
            }
        ],
        agg_funcs=["count"],
    )

    # Execute
    start_time = time.time()
    config = pz.QueryProcessorConfig(
        policy=policy,
        verbose=args.verbose,
        execution_strategy="sequential",
        available_models=[pz.Model.GPT_5],
    )
    data_record_collection = grouped.run(config)
    exec_time = time.time() - start_time

    # Post-process: rename the semantic group key to "era"
    result_df = pd.DataFrame([
        {
            "era": r.releaseDateTheaters,
            "review_count": r.reviewId,
        }
        for r in data_record_collection
    ])
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    result_df.to_csv(args.output, index=False)

    if args.stats_output is not None:
        os.makedirs(os.path.dirname(args.stats_output) or ".", exist_ok=True)
        with open(args.stats_output, "w") as f:
            json.dump(data_record_collection.execution_stats.to_json(), f, indent=2)

    print(f"\nExecution time: {exec_time:.2f}s")
    print(f"Results saved to: {args.output}")
    if args.stats_output is not None:
        print(f"Execution stats saved to: {args.stats_output}")
    print(f"Generated {len(result_df)} era groups")


if __name__ == "__main__":
    main()
