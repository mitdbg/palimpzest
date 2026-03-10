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

    # Hierarchical semantic groupby:
    #   Level 0 — infer primary movie genre from reviewText (constrained to 11 values)
    #   Level 1 — group by the existing isTopCritic boolean field
    groupby_fields = [
        [
            {
                "name": "reviewText",
                "type": str,
                "desc": (
                    "The primary genre of the movie being reviewed, inferred from the review text. "
                    "Must be exactly one of these values — no other labels are allowed: "
                    "'Action', 'Adventure', 'Comedy', 'Crime', 'Documentary', "
                    "'Drama', 'History', 'Mystery & thriller', 'Romance', 'Sci-fi', 'War'."
                ),
            }
        ],
        [
            {
                "name": "isTopCritic",
                "type": str,
                "desc": (
                    "Whether the reviewer is a top critic. "
                    "Use the existing isTopCritic field value directly — "
                    "True maps to 'Top Critic', False maps to 'Not Top Critic'. "
                    "Do not use any other labels."
                ),
            }
        ],
    ]
    agg_fields = [
        [{"name": "scoreSentiment", "type": str, "desc": "Sentiment label for the review"}],
        [{"name": "scoreSentiment", "type": str, "desc": "Sentiment label for the review"}],
    ]
    agg_funcs = [
        ["list"],
        ["list"]
    ]

    start_time = time.time()
    # hierarchical_sem_groupby now returns (nested_result, accumulated_gen_stats)
    nested_result, gen_stats = reviews.hierarchical_sem_groupby(
        groupby_fields=groupby_fields,
        agg_fields=agg_fields,
        agg_funcs=agg_funcs
    )
    exec_time = time.time() - start_time

    # Flatten nested results and compute frac_positive
    rows = []
    for genre_key, inner_result in nested_result.items():
        genre = genre_key[0] if isinstance(genre_key, tuple) else genre_key
        for r in inner_result.data_records:
            # Normalize LLM string → boolean to match the GT's isTopCritic format
            raw_itc = str(r.isTopCritic).strip().lower()
            is_top_critic = raw_itc in ("top critic", "true", "yes", "1")
            sentiments = r.scoreSentiment
            frac_pos = (
                sum(1 for s in sentiments if str(s).upper() == "POSITIVE") / len(sentiments)
                if len(sentiments) > 0 else 0.0
            )
            rows.append({
                "primaryGenre": genre,
                "isTopCritic": is_top_critic,
                "frac_positive": frac_pos,
                "review_count": len(sentiments)
            })
    result_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    result_df.to_csv(args.output, index=False)

    if args.stats_output is not None:
        os.makedirs(os.path.dirname(args.stats_output) or ".", exist_ok=True)
        total_cost = gen_stats.total_input_cost + gen_stats.total_output_cost
        total_tokens = int(gen_stats.total_input_tokens + gen_stats.total_output_tokens)
        stats = {
            "total_execution_time": exec_time,
            "total_execution_cost": total_cost,
            "total_tokens": total_tokens,
            "optimization_time": 0.0,
            "plan_execution_time": exec_time,
        }
        with open(args.stats_output, "w") as f:
            json.dump(stats, f, indent=2)

    print(f"\nExecution time: {exec_time:.2f}s")
    print(f"Total cost: ${gen_stats.total_input_cost + gen_stats.total_output_cost:.4f}")
    print(f"Total tokens: {int(gen_stats.total_input_tokens + gen_stats.total_output_tokens):,}")
    print(f"Results saved to: {args.output}")
    if args.stats_output is not None:
        print(f"Execution stats saved to: {args.stats_output}")
    print(f"Generated {len(result_df)} genre-topcritic groups")


if __name__ == "__main__":
    main()
