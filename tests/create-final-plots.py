#!/usr/bin/env python3
from palimpzest.profiler import Profiler, StatsProcessor
import palimpzest as pz

from palimpzest.execution import graphicEmit, flatten_nested_tuples
from palimpzest.operators import InduceFromCandidateOp

from PIL import Image
from sklearn.metrics import precision_recall_fscore_support
from tabulate import tabulate

import matplotlib.pyplot as plt
import pandas as pd

import argparse
import json
import shutil
import subprocess
import time
import os
import pdb





def plot_runtime_cost_vs_quality(workload_results, opt):
    # create figure
    fig_text, axs_text = plt.subplots(nrows=2, ncols=3, figsize=(15,6))
    fig_clean, axs_clean = plt.subplots(nrows=2, ncols=3, figsize=(15,6))

    workload_to_col = {"enron": 0, "real-estate": 1, "biofabric": 2}

    # parse results into fields
    for workload, results in workload_results.items():
        col = workload_to_col[workload]

        pareto_indices = []
        for i, result_dict in results:
            totalTime_i, totalCost_i, quality_i = result_dict["runtime"], result_dict["cost"], result_dict["f1_score"]
            paretoFrontier = True

            # check if any other plan dominates plan i
            for j, _result_dict in results:
                totalTime_j, totalCost_j, quality_j = _result_dict["runtime"], _result_dict["cost"], _result_dict["f1_score"]
                if i == j:
                    continue

                # if plan i is dominated by plan j, set paretoFrontier = False and break
                if totalTime_j <= totalTime_i and totalCost_j <= totalCost_i and quality_j >= quality_i:
                    paretoFrontier = False
                    break

            # add plan i to pareto frontier if it's not dominated
            if paretoFrontier:
                pareto_indices.append(i)

        for plan_idx, result_dict in results:
            runtime = result_dict["runtime"]
            cost = result_dict["cost"]
            f1_score = result_dict["f1_score"]
            text = plan_idx

            # set label and color
            color = "black"
            marker = None # if plan_idx not in pareto_indices or f1_score == 0.0 else '*'
            s = None # if plan_idx not in pareto_indices or f1_score == 0.0 else 200.0

            # plot runtime vs. f1_score and cost vs. f1_score
            axs_text[0][col].scatter(f1_score, runtime, alpha=0.6, color=color, marker=marker, s=s)
            axs_text[1][col].scatter(f1_score, cost, alpha=0.6, color=color, marker=marker, s=s)
            axs_clean[0][col].scatter(f1_score, runtime, alpha=0.6, color=color,marker=marker, s=s)
            axs_clean[1][col].scatter(f1_score, cost, alpha=0.6, color=color, marker=marker, s=s)

            # add annotations
            axs_text[0][col].annotate(text, (f1_score, runtime))
            axs_text[1][col].annotate(text, (f1_score, cost))

        # set x,y-lim for each workload
        left, right = -0.05, 1.05
        if workload == "real-estate":
            left = 0.5
            right = 0.85
        elif workload == "biofabric":
            left = 0.3
            right = 0.6
        axs_text[0][col].set_xlim(left, right)
        axs_text[1][col].set_xlim(left, right)
        axs_clean[0][col].set_xlim(left, right)
        axs_clean[1][col].set_xlim(left, right)

        # turn on grid lines
        axs_text[0][col].grid(True, alpha=0.4)
        axs_text[1][col].grid(True, alpha=0.4)
        axs_clean[0][col].grid(True, alpha=0.4)
        axs_clean[1][col].grid(True, alpha=0.4)

    # remove ticks from first row
    for idx in range(3):
        axs_text[0][idx].set_xticklabels([])
        axs_clean[0][idx].set_xticklabels([])

    # savefigs
    workload_to_title = {
        "enron": "Legal Discovery",
        "real-estate": "Real Estate Search",
        "biofabric": "Medical Schema Matching"
    }
    for workload, title in workload_to_title.items():
        idx = 0 if workload == "enron" else (1 if workload == "real-estate" else 2)
        axs_text[0][idx].set_title(f"{title}", fontsize=12)
        axs_clean[0][idx].set_title(f"{title}", fontsize=12)

    axs_text[0][0].set_ylabel("Runtime (seconds)", fontsize=12)
    axs_text[1][0].set_ylabel("Cost (USD)", fontsize=12)
    for idx in range(3):
        axs_text[1][idx].set_xlabel("F1 Score", fontsize=12)
    
    axs_clean[0][0].set_ylabel("Runtime (seconds)", fontsize=12)
    axs_clean[1][0].set_ylabel("Cost (USD)", fontsize=12)
    for idx in range(3):
        axs_clean[1][idx].set_xlabel("F1 Score", fontsize=12)

    fig_text.savefig(f"final-eval-results/plots/{opt}-text-bw.png", dpi=500, bbox_inches="tight")
    fig_clean.savefig(f"final-eval-results/plots/{opt}-clean-bw.png", dpi=500, bbox_inches="tight")



if __name__ == "__main__":
    # parse arguments
    startTime = time.time()
    parser = argparse.ArgumentParser(description='Run the evaluation(s) for the paper')
    parser.add_argument('--workload', type=str, help='The workload: one of ["biofabric", "enron", "real-estate"]')
    parser.add_argument('--opt' , type=str, help='The optimization: one of ["model", "codegen", "token-reduction"]')
    parser.add_argument('--reoptimize', default=False, action='store_true', help='Run reoptimization')

    args = parser.parse_args()

    # create directory for intermediate results
    os.makedirs(f"final-eval-results/plots", exist_ok=True)

    if args.reoptimize:
        plot_reoptimization(args.workload)
        exit(1)

    # opt and workload to # of plots
    opt_workload_to_num_plans = {
        "model": {
            "enron": 21,
            "real-estate": 21,
            "biofabric": 14,
        },
        "codegen": {
            "enron": 8,
            "real-estate": 12,
            "biofabric": 6,
        },
        "token-reduction": {
            "enron": 16,
            "real-estate": 24,
            "biofabric": 16,
        },
    }

    # read results file(s) generated by evaluate_pz_plans
    results = {"enron": [], "real-estate": [], "biofabric": []}
    for workload in ["enron", "real-estate", "biofabric"]:
        num_plans = opt_workload_to_num_plans[args.opt][workload]
        for plan_idx in range(num_plans):
            # if workload == "real-estate" and args.opt in ["models", "token-reduction"] and plan_idx == 9:
            #     continue

            if workload == "biofabric" and args.opt == "codegen" and plan_idx == 3:
                continue

            with open(f"final-eval-results/{args.opt}/{workload}/results-{plan_idx}.json", 'r') as f:
                result = json.load(f)
                results[workload].append((plan_idx, result))

    plot_runtime_cost_vs_quality(results, args.opt)
