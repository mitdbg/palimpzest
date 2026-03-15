#!/usr/bin/env python3
"""
Query 1 — Sentiment by Publication (Palimpzest)

Group by publicatioName and compute the fraction of positive reviews.

Pipeline:
  1. sem_groupby – Semantically groups the records by `publicatioName`
                   (the LLM normalises slight variations in publication
                   names) and collects the scoreSentiment values into a
                   list per group.
  2. Post-processing – computes frac_positive from the collected lists.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Add the src directory to the path
repo_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(repo_root / "src"))

import palimpzest as pz

load_dotenv()


def compute_frac_positive(sentiments):
    """Compute fraction of positive sentiments from a collected list."""
    num_pos = sum(1 for s in sentiments if s and str(s).upper() == "POSITIVE")
    total = len(sentiments)
    return num_pos / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Query 1: Sentiment by Publication")
    parser.add_argument("--verbose", default=False, action="store_true")
    parser.add_argument("--policy", type=str, default="maxquality",
                       help="One of 'mincost', 'mintime', 'maxquality'")
    parser.add_argument("--output", type=str, default="query1_pz_output.csv")
    parser.add_argument("--stats-output", type=str, default=None,
                        help="Optional path to write execution stats JSON")
    parser.add_argument(
        "--execution-strategy",
        type=str,
        default="sequential",
        help="One of 'sequential', 'pipelined', 'parallel'",
    )
    args = parser.parse_args()

    # Set policy
    policy_map = {
        "mincost": pz.MinCost(),
        "mintime": pz.MinTime(),
        "maxquality": pz.MaxQuality()
    }
    policy = policy_map.get(args.policy, pz.MaxQuality())

    # Load data
    script_dir = Path(__file__).parent
    csv_path = script_dir / "../movie_reviews.csv"
    print(f"Loading reviews from: {csv_path}")

    csv_df = pd.read_csv(csv_path).head(500)
    print(f"Loaded {len(csv_df)} reviews")

    # ── Ingest the DataFrame ─────────────────────────────────────────
    # MemoryDataset automatically creates a schema from the DataFrame.
    # The CSV already contains: publicatioName, reviewText,
    # scoreSentiment, etc.
    reviews = pz.MemoryDataset(id="reviews", vals=csv_df)

    # ── sem_groupby – semantically group by publication name ─────────
    # The LLM normalises publication names (e.g. "NY Times" vs
    # "The New York Times") and groups the records accordingly.
    # We collect the existing scoreSentiment values into a list per
    # group so we can compute the fraction of positive reviews.
    grouped = reviews.sem_groupby(
        gby_fields=["publicatioName"],
        agg_fields=["scoreSentiment"],
        agg_funcs=["list"],
    )

    # ── Execute ───────────────────────────────────────────────────────
    start_time = time.time()
    config = pz.QueryProcessorConfig(
        policy=policy,
        verbose=args.verbose,
        execution_strategy=args.execution_strategy,
    )
    data_record_collection = grouped.run(config)
    exec_time = time.time() - start_time

    # ── Post-process – compute frac_positive per group ────────────────
    result_df = pd.DataFrame([
        {
            "publicatioName": r.publicatioName,
            "frac_positive": compute_frac_positive(
                getattr(r, "scoreSentiment", []) or []
            ),
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
    print(f"Generated {len(result_df)} publication groups")


if __name__ == "__main__":
    main()
