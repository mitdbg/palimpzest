#!/usr/bin/env python3
"""
Query 5 — Emotional Tone by Director and Genre (Palimpzest)

Finer-grained emotional tone classification beyond binary sentiment.

Pipeline:
  1. Join movie_reviews with movies filtered by director + genre.
  2. sem_groupby – LLM reads reviewText and groups by emotional tone
     (Enthusiastic, Measured, Disappointed); counts reviewId per group.
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
    parser = argparse.ArgumentParser(description="Query 5: Emotional Tone by Director and Genre")
    parser.add_argument("--director", type=str, default="Steven Spielberg")
    parser.add_argument("--genre", type=str, default="Adventure")
    parser.add_argument("--verbose", default=False, action="store_true")
    parser.add_argument("--policy", type=str, default="maxquality")
    parser.add_argument("--output", type=str, default="query5_pz_output.csv")
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

    filtered_movies = movies_df[
        movies_df["director"].str.contains(args.director, na=False, case=False)
        & movies_df["genre"].str.contains(args.genre, na=False, case=False)
    ][["id"]]

    merged_df = reviews_df.merge(filtered_movies, on="id", how="inner")
    print(f"Loaded {len(merged_df)} reviews for {args.director} in {args.genre}")

    reviews = pz.MemoryDataset(id="reviews", vals=merged_df)

    # sem_groupby: LLM reads reviewText and groups by emotional tone, count reviewId
    grouped = reviews.sem_groupby(
        gby_fields=["reviewText"],
        agg_fields=["reviewId"],
        agg_funcs=["count"],
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

    # Post-process: rename the semantic group key to "emotionalTone"
    result_df = pd.DataFrame([
        {
            "emotionalTone": r.reviewText,
            "review_count": r.reviewId,
            "director": args.director,
            "genre": args.genre,
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
    print(f"Generated {len(result_df)} tone groups")


if __name__ == "__main__":
    main()
