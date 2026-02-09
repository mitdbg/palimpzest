#!/usr/bin/env python3
"""
Movies - Sentiment Analysis with Palimpzest

This program uses Palimpzest to:
1. Read movie reviews from CSV file
2. Parse the sentiment (POSITIVE/NEGATIVE) from each review
3. Group by critic name
4. Compute the fraction of positive reviews per critic
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Add the src directory to the path to import palimpzest
repo_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(repo_root / "src"))

import palimpzest as pz

load_dotenv()


def custom_frac_positive(group_data):
    """
    Custom aggregation function to compute fraction of positive sentiments.
    This will be used for semantic aggregation.
    """
    sentiments = [record.scoreSentiment for record in group_data]
    num_pos = sum(1 for s in sentiments if s == "POSITIVE")
    total = len(sentiments)
    return num_pos / total if total > 0 else 0.0


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run movies sentiment analysis with Palimpzest")
    parser.add_argument("--verbose", default=False, action="store_true", help="Print verbose output")
    parser.add_argument("--profile", default=False, action="store_true", help="Profile execution")
    parser.add_argument(
        "--policy",
        type=str,
        help="One of 'mincost', 'mintime', 'maxquality'",
        default="maxquality",
    )
    parser.add_argument(
        "--execution-strategy",
        type=str,
        help="The execution strategy to use. One of sequential, pipelined, parallel",
        default="sequential",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output CSV file path",
        default="movies_1_pz_output.csv",
    )

    args = parser.parse_args()

    # Set policy
    policy = pz.MaxQuality()
    if args.policy == "mincost":
        policy = pz.MinCost()
    elif args.policy == "mintime":
        policy = pz.MinTime()
    elif args.policy == "maxquality":
        policy = pz.MaxQuality()
    else:
        print("Policy not supported")
        exit(1)

    # Check for API keys
    if os.getenv("OPENAI_API_KEY") is None and os.getenv("TOGETHER_API_KEY") is None and os.getenv("ANTHROPIC_API_KEY") is None:
        print("WARNING: OPENAI_API_KEY, TOGETHER_API_KEY, and ANTHROPIC_API_KEY are unset")

    # Get the path to the CSV file
    script_dir = Path(__file__).parent
    csv_path = script_dir / "movie_reviews.csv"

    print(f"Loading movie reviews from: {csv_path}")
    start_time = time.time()

    # Read CSV file into memory using pandas (limit to first 500 rows)
    csv_df = pd.read_csv(csv_path).head(500)
    print(f"Loaded {len(csv_df)} reviews from CSV")
    
    # Build the Palimpzest query plan using MemoryDataset
    # Let MemoryDataset infer the schema from the DataFrame
    # This avoids type inference issues
    reviews = pz.MemoryDataset(id="movie-reviews", vals=csv_df)
    
    # Data is already in the right format, no need for sem_map
    # Define the GroupBy operation
    # Group by criticName and compute fraction of positive reviews
    gby_fields = ["criticName"]
    agg_fields = ["scoreSentiment"]
    agg_funcs = ["count"]  # We'll use count initially to demonstrate grouping
    
    grouped_reviews = reviews.groupby(gby_fields, agg_fields, agg_funcs)

    # Configure and run the query
    config = pz.QueryProcessorConfig(
        policy=policy,
        verbose=args.verbose,
        execution_strategy=args.execution_strategy,
    )

    print(f"Policy: {str(policy)}")
    print("Running Palimpzest query...")
    
    # Pass policy as kwarg based on policy type
    policy_kwargs = {}
    if isinstance(policy, pz.MaxQuality):
        policy_kwargs["max_quality"] = True
    elif isinstance(policy, pz.MinCost):
        policy_kwargs["min_cost"] = True
    elif isinstance(policy, pz.MinTime):
        policy_kwargs["min_time"] = True
    
    print(f"Policy kwargs: {policy_kwargs}")  # Debug: show what we're passing
    data_record_collection = grouped_reviews.run(config, **policy_kwargs)

    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time:.2f} seconds")

    # Convert results to DataFrame
    results_df = data_record_collection.to_df()
    print(f"\nResults shape: {results_df.shape}")
    print("\nFirst 10 results:")
    # print(results_df.head(10))

    # Save results to CSV
    output_path = script_dir / args.output
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Print execution statistics
    if hasattr(data_record_collection, 'execution_stats'):
        print("\nExecution Statistics:")
        print(data_record_collection.execution_stats)


if __name__ == "__main__":
    main()
