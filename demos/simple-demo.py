#!/usr/bin/env python3
import argparse
import os
import time

from demo_core import execute_task, format_results_table

from palimpzest.policy import MaxQuality, MinCost, MinTime


def main():
    # parse arguments
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Run a simple demo")
    parser.add_argument("--verbose", default=False, action="store_true", help="Print verbose output")
    parser.add_argument("--profile", default=False, action="store_true", help="Profile execution")
    parser.add_argument("--datasetid", type=str, help="The dataset id")
    parser.add_argument("--task", type=str, help="The task to run")
    parser.add_argument(
        "--execution_strategy",
        type=str,
        help="The execution strategy to use. One of sequential, pipelined_parallel, pipelined_single_thread",
        default="sequential",
    )
    parser.add_argument(
        "--policy",
        type=str,
        help="One of 'mincost', 'mintime', 'maxquality'",
        default="mincost",
    )

    args = parser.parse_args()

    # The user has to indicate the dataset id and the task
    if args.datasetid is None:
        print("Please provide a dataset id")
        exit(1)
    if args.task is None:
        print("Please provide a task")
        exit(1)

    # Set up execution parameters
    datasetid = args.datasetid
    task = args.task
    verbose = args.verbose
    profile = args.profile

    # Set policy
    policy = MaxQuality()
    if args.policy == "mincost":
        policy = MinCost()
    elif args.policy == "mintime":
        policy = MinTime()
    elif args.policy == "maxquality":
        policy = MaxQuality()
    else:
        print("Policy not supported for this demo")
        exit(1)

    if os.getenv("OPENAI_API_KEY") is None and os.getenv("TOGETHER_API_KEY") is None:
        print("WARNING: Both OPENAI_API_KEY and TOGETHER_API_KEY are unset")

    # Execute task
    records, execution_stats, cols = execute_task(
        task=task,
        datasetid=datasetid,
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
