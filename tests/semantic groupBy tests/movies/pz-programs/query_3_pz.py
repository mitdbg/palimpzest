#!/usr/bin/env python3
"""
Query 3 — Fraction Positive per Audience Type (Palimpzest)

For a specific director, group reviews by MPAA-inferred audience type
and compute fraction positive.

Pipeline:
  1. Join movie_reviews with movies filtered by director to get rating.
  2. sem_groupby – LLM semantically normalises the MPAA rating into
     audience-type buckets (Children, Teen, Adult, Unrated); lists
     scoreSentiment per group.
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
    parser = argparse.ArgumentParser(description="Query 3: Sentiment by Audience Type")
    parser.add_argument("--director", type=str, default="Christopher Nolan",
                        help="Director name to filter by")
    parser.add_argument("--verbose", default=False, action="store_true")
    parser.add_argument("--policy", type=str, default="maxquality")
    parser.add_argument("--output", type=str, default="query3_pz_output.csv")
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

    # Load and filter data
    reviews_df = pd.read_csv(script_dir / "../movie_reviews.csv").head(500)
    movies_df = pd.read_csv(script_dir / "../movies.csv")

    # Filter for director's movies and keep the rating column
    director_movies = movies_df[
        movies_df["director"].str.contains(args.director, na=False, case=False)
    ][["id", "rating"]]

    merged_df = reviews_df.merge(director_movies, on="id", how="inner")
    print(f"Loaded {len(merged_df)} reviews for {args.director}")

    reviews = pz.MemoryDataset(id="reviews", vals=merged_df)

    # sem_groupby: LLM maps MPAA rating → audience type bucket, list scoreSentiment
    grouped = reviews.sem_groupby(
        gby_fields=["rating"],
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
            "audienceType": r.rating,
            "frac_positive": (
                sum(1 for s in r.scoreSentiment if str(s).upper() == "POSITIVE")
                / len(r.scoreSentiment)
                if len(r.scoreSentiment) > 0
                else 0.0
            ),
            "review_count": len(r.scoreSentiment),
            "director": args.director,
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
    print(f"Generated {len(result_df)} audience type groups")


if __name__ == "__main__":
    main()
