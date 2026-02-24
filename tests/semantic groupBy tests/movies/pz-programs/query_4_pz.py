#!/usr/bin/env python3
"""
Query 4 — Sentiment and Top Critic Bias by Genre (Palimpzest)

Hard query: genre must be inferred from review text itself (not available
in reviews table). Both group key and aggregation value are semantic.

Pipeline:
  1. Load movie_reviews.
  2. sem_groupby – LLM infers primaryGenre from reviewText and groups by
     [primaryGenre, isTopCritic]; lists scoreSentiment per group.
  3. Post-process list → frac_positive.
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
    parser = argparse.ArgumentParser(description="Query 4: Sentiment by Inferred Genre")
    parser.add_argument("--verbose", default=False, action="store_true")
    parser.add_argument("--policy", type=str, default="maxquality")
    parser.add_argument("--output", type=str, default="query4_pz_output.csv")
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

    reviews_df = pd.read_csv(script_dir / "../movie_reviews.csv").head(500)
    print(f"Loaded {len(reviews_df)} reviews")

    reviews = pz.MemoryDataset(id="reviews", vals=reviews_df)

    # sem_groupby: LLM infers primaryGenre from reviewText,
    #              groups by [reviewText (→ genre), isTopCritic],
    #              lists scoreSentiment per group.
    grouped = reviews.sem_groupby(
        gby_fields=["reviewText", "isTopCritic"],
        agg_fields=["scoreSentiment"],
        agg_funcs=["list"],
    )

    # Execute
    start_time = time.time()
    config = pz.QueryProcessorConfig(
        policy=policy,
        verbose=args.verbose,
        execution_strategy=args.execution_strategy,
    )
    data_record_collection = grouped.run(config)
    exec_time = time.time() - start_time

    # Post-process: compute frac_positive from the sentiment lists
    result_df = pd.DataFrame([
        {
            "primaryGenre": r.reviewText,
            "isTopCritic": r.isTopCritic,
            "frac_positive": (
                sum(1 for s in r.scoreSentiment if str(s).upper() == "POSITIVE")
                / len(r.scoreSentiment)
                if len(r.scoreSentiment) > 0
                else 0.0
            ),
            "review_count": len(r.scoreSentiment),
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
    print(f"Generated {len(result_df)} genre-topcritic groups")


if __name__ == "__main__":
    main()
