#!/usr/bin/env python3
"""
Query 6 — Most Positive Review by Director (Palimpzest)

Pipeline:
  1. Join movie_reviews with movies to get director per review.
  2. Drop records with missing or unparseable originalScore; normalise to [0, 1].
  3. Python groupby("director") — exact, non-semantic.
  4. For each director group: sem_map to score each review's positivity.
  5. Find the review with the highest positivity score.

Comparison metric: |ground_truth_normalizedScore − pz_normalizedScore| per director.
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


def parse_score(score_str) -> float | None:
    """Parse "3.5/4", "4/5", etc. into a float in [0, 1]. Returns None if unparseable."""
    if pd.isna(score_str) or str(score_str).strip() == "":
        return None
    parts = str(score_str).strip().split("/")
    if len(parts) == 2:
        try:
            num, den = float(parts[0]), float(parts[1])
            return num / den if den != 0 else None
        except ValueError:
            return None
    return None


def main():
    parser = argparse.ArgumentParser(description="Query 6: Most Positive Review by Director")
    parser.add_argument("--verbose", default=False, action="store_true")
    parser.add_argument("--policy", type=str, default="maxquality",
                        help="One of 'mincost', 'mintime', 'maxquality'")
    parser.add_argument("--output", type=str, default="query6_pz_output.csv")
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

    # ── Load and prepare data ─────────────────────────────────────────
    reviews_df = pd.read_csv(script_dir / "../movie_reviews.csv")
    movies_df  = pd.read_csv(script_dir / "../movies.csv")[["id", "director"]]

    merged_df = reviews_df.merge(movies_df, on="id", how="left")
    merged_df = merged_df.dropna(subset=["originalScore"])
    merged_df = merged_df[merged_df["originalScore"].str.strip() != ""]
    merged_df["normalizedScore"] = merged_df["originalScore"].apply(parse_score)
    merged_df = merged_df.dropna(subset=["normalizedScore", "director"])

    directors = merged_df["director"].unique()
    print(f"Loaded {len(merged_df)} reviews across {len(directors)} directors")

    # ── Non-semantic groupby + sem_agg per director ───────────────────
    rows = []
    # Accumulated execution stats across all sem_agg calls
    acc_input_tokens  = 0
    acc_output_tokens = 0
    acc_exec_cost     = 0.0
    acc_opt_time      = 0.0
    acc_plan_time     = 0.0

    wall_start = time.time()

    count = 0
    for director, group_df in merged_df.groupby("director"):
        if count >= 40:
            break 

        count += 1
        # Keep the full group_df for lookup later
        full_group_df = group_df[["reviewText", "normalizedScore"]].reset_index(drop=True)
        
        # Build a PZ dataset with reviewText and normalizedScore
        ds = pz.MemoryDataset(id="reviews", vals=full_group_df)

        # Use sem_map to score each review's positivity (0-10 scale)
        scored_ds = ds.sem_map(
            cols=[{
                "name": "positivityScore",
                "type": float,
                "desc": "A score from 0 to 10 indicating how positive this review is, where 10 is extremely positive and 0 is very negative.",
            }],
            depends_on="reviewText",
        )

        # Create fresh config for each director group
        config = pz.QueryProcessorConfig(
            policy=policy,
            verbose=args.verbose,
            execution_strategy=args.execution_strategy,
        )
        
        result_collection = scored_ds.run(config)

        # Find the review with the highest positivity score
        max_score_idx = -1
        max_positivity = -1
        scored_reviews = []
        for idx, r in enumerate(result_collection):
            scored_reviews.append(r)
            if r.positivityScore > max_positivity:
                max_positivity = r.positivityScore
                max_score_idx = idx
        
        # Get the most positive review
        most_positive = None
        norm_score = None
        if max_score_idx >= 0:
            best_review = scored_reviews[max_score_idx]
            most_positive = best_review.reviewText
            norm_score = float(best_review.normalizedScore)

        rows.append({
            "director":           director,
            "mostPositiveReview": most_positive,
            "normalizedScore":    norm_score,
        })

        # Accumulate execution stats from this director's run
        es = result_collection.execution_stats
        acc_input_tokens  += es.total_input_tokens
        acc_output_tokens += es.total_output_tokens
        acc_exec_cost     += es.total_execution_cost
        acc_opt_time      += es.optimization_time
        acc_plan_time     += es.plan_execution_time

    wall_time = time.time() - wall_start

    # ── Save results ──────────────────────────────────────────────────
    result_df = pd.DataFrame(rows).sort_values("director").reset_index(drop=True)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    result_df.to_csv(args.output, index=False)

    # ── Save execution stats ──────────────────────────────────────────
    if args.stats_output is not None:
        stats = {
            "total_execution_time":    wall_time,
            "total_optimization_time": acc_opt_time,
            "plan_execution_time":     acc_plan_time,
            "total_input_tokens":      acc_input_tokens,
            "total_output_tokens":     acc_output_tokens,
            "total_tokens":            acc_input_tokens + acc_output_tokens,
            "total_execution_cost":    acc_exec_cost,
            "num_directors":           len(rows),
        }
        os.makedirs(os.path.dirname(args.stats_output) or ".", exist_ok=True)
        with open(args.stats_output, "w") as f:
            json.dump(stats, f, indent=2)

    print(f"\nExecution time:   {wall_time:.2f}s")
    print(f"Total tokens:     {acc_input_tokens + acc_output_tokens:,}")
    print(f"Total cost:       ${acc_exec_cost:.4f}")
    print(f"Results saved to: {args.output}")
    if args.stats_output is not None:
        print(f"Execution stats saved to: {args.stats_output}")
    print(f"Generated {len(result_df)} director groups")


if __name__ == "__main__":
    main()
