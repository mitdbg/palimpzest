#!/usr/bin/env python3
import argparse
import logging
import os
import time

from demo_core import execute_task, format_results_table
from dotenv import load_dotenv

import palimpzest as pz
from palimpzest.tools.logger import setup_logger

load_dotenv()

def main():
    # parse arguments
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Run a simple demo")
    parser.add_argument("--verbose", default=False, action="store_true", help="Print verbose output")
    parser.add_argument("--profile", default=False, action="store_true", help="Profile execution")
    parser.add_argument("--dataset", type=str, help="Path to the dataset")
    parser.add_argument("--task", type=str, help="The task to run")
    parser.add_argument(
        "--execution_strategy",
        type=str,
        help="The execution strategy to use. One of sequential, pipelined, parallel",
        default="sequential",
    )
    parser.add_argument(
        "--policy",
        type=str,
        help="One of 'mincost', 'mintime', 'maxquality'",
        default="mincost",
    )

    args = parser.parse_args()

    # The user has to indicate the dataset and the task
    if args.dataset is None:
        print("Please provide a path to the dataset")
        exit(1)
    if args.task is None:
        print("Please provide a task")
        exit(1)

    # Set up execution parameters
    dataset = args.dataset
    task = args.task
    verbose = args.verbose
    profile = args.profile

    # Set policy
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

    # Execute task
    logger = setup_logger("palimpzest")
    logger.pz_logger.set_console_level(logging.DEBUG if verbose else logging.ERROR)
    records, execution_stats, cols = execute_task(
        task=task,
        dataset=dataset,
        policy=policy,
        verbose=verbose,
        profile=profile,
        execution_strategy=args.execution_strategy
    )

    # Print results
    print(f"Policy is: {str(policy)}")
    print("Executed plan:")
    plan_str = list(execution_stats.plan_strs.values())[0]
    print(plan_str)
    end_time = time.time()
    print("Elapsed time:", end_time - start_time)

    print(format_results_table(records, cols=cols))

if __name__ == "__main__":
    main()
