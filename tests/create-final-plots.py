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
import seaborn as sns

import argparse
import json
import shutil
import subprocess
import time
import os
import pdb


def get_color(opt, workload, result_dict):
    opt_to_color = {"model": "#87bc45", "codegen": "#27aeef", "token-reduction": "#b33dc6"}

    color = "black"
    if opt == "model" and len(set(filter(None, result_dict['plan_info']['models']))) > 1:
        color = opt_to_color[opt]

    elif opt == "codegen" and "codegen-with-fallback" in result_dict['plan_info']['query_strategies']:
        color = opt_to_color[opt]

    elif opt == "token-reduction" and any([budget is not None and budget < 1.0 for budget in result_dict['plan_info']['token_budgets']]):
        color = opt_to_color[opt]

    elif workload == "real-estate":
        # give color to logical re-orderings on real-estate
        if result_dict['plan_info']['models'][1] == "gpt-4-vision-preview":
            color = opt_to_color[opt]
    
    return color


def get_pareto_indices(result_dicts, col):
    pareto_indices = []
    for idx, (_, result_dict) in enumerate(result_dicts):
        col_i, quality_i = result_dict[col], result_dict["f1_score"]
        paretoFrontier = True

        # check if any other plan dominates plan i
        for j, (_, _result_dict) in enumerate(result_dicts):
            col_j, quality_j = _result_dict[col], _result_dict["f1_score"]
            if idx == j:
                continue

            # if plan i is dominated by plan j, set paretoFrontier = False and break
            if col_j <= col_i and quality_j >= quality_i:
                paretoFrontier = False
                break

        # add plan i to pareto frontier if it's not dominated
        if paretoFrontier:
            pareto_indices.append(idx)

    return pareto_indices


def plot_runtime_cost_vs_quality(results):
    # create figure
    fig_text, axs_text = plt.subplots(nrows=2, ncols=3, figsize=(15,6))
    fig_clean, axs_clean = plt.subplots(nrows=2, ncols=3, figsize=(15,6))
    fig_clean_bw, axs_clean_bw = plt.subplots(nrows=2, ncols=3, figsize=(15,6))
    fig_clean_mc, axs_clean_mc = plt.subplots(nrows=2, ncols=3, figsize=(15,6))

    workload_to_col = {"enron": 0, "real-estate": 1, "biofabric": 2}

    # parse results into fields
    for workload, opt_results in results.items():
        col = workload_to_col[workload]

        all_result_dicts = []        
        for opt, result_dicts in opt_results.items():
            all_result_dicts.extend(result_dicts)

            for plan_idx, result_dict in result_dicts:
                runtime = result_dict["runtime"]
                cost = result_dict["cost"]
                f1_score = result_dict["f1_score"]

                text = f"{opt[0]}{plan_idx}"

                # set label and color
                color = get_color(opt, workload, result_dict)
                marker = 'D' if color == "black" else None
                mcolor = "black" if color == "black" else "#87bc45"

                # plot runtime vs. f1_score and cost vs. f1_score
                axs_text[0][col].scatter(f1_score, runtime, alpha=0.6, color=color)
                axs_text[1][col].scatter(f1_score, cost, alpha=0.6, color=color)
                axs_clean[0][col].scatter(f1_score, runtime, alpha=0.6, color=color)
                axs_clean[1][col].scatter(f1_score, cost, alpha=0.6, color=color)
                axs_clean_bw[0][col].scatter(f1_score, runtime, alpha=0.6, color="black", marker=marker)
                axs_clean_bw[1][col].scatter(f1_score, cost, alpha=0.6, color="black", marker=marker)
                axs_clean_mc[0][col].scatter(f1_score, runtime, alpha=0.6, color=mcolor, marker=marker)
                axs_clean_mc[1][col].scatter(f1_score, cost, alpha=0.6, color=mcolor, marker=marker)

                # add annotations
                axs_text[0][col].annotate(text, (f1_score, runtime))
                axs_text[1][col].annotate(text, (f1_score, cost))

        # compute pareto frontiers across all optimizations
        cost_pareto_lst_indices = get_pareto_indices(all_result_dicts, "cost")
        runtime_pareto_lst_indices = get_pareto_indices(all_result_dicts, "runtime")

        # plot line for pareto frontiers
        cost_pareto_qualities = [all_result_dicts[idx][1]['f1_score'] for idx in cost_pareto_lst_indices]
        pareto_costs = [all_result_dicts[idx][1]['cost'] for idx in cost_pareto_lst_indices]
        cost_pareto_curve = zip(cost_pareto_qualities, pareto_costs)
        cost_pareto_curve = sorted(cost_pareto_curve, key=lambda tup: tup[0])
        if workload == "biofabric":
            cost_pareto_curve = list(filter(lambda tup: tup[0] > 0.3, cost_pareto_curve))
        pareto_cost_xs, pareto_cost_ys = zip(*cost_pareto_curve)

        runtime_pareto_qualities = [all_result_dicts[idx][1]['f1_score'] for idx in runtime_pareto_lst_indices]
        pareto_runtimes = [all_result_dicts[idx][1]['runtime'] for idx in runtime_pareto_lst_indices]
        runtime_pareto_curve = zip(runtime_pareto_qualities, pareto_runtimes)
        runtime_pareto_curve = sorted(runtime_pareto_curve, key=lambda tup: tup[0])
        if workload == "biofabric":
            runtime_pareto_curve = list(filter(lambda tup: tup[0] > 0.3, runtime_pareto_curve))
        pareto_runtime_xs, pareto_runtime_ys = zip(*runtime_pareto_curve)

        axs_text[0][col].plot(pareto_runtime_xs, pareto_runtime_ys, color="#ef9b20", linestyle='--')
        axs_text[1][col].plot(pareto_cost_xs, pareto_cost_ys, color="#ef9b20", linestyle='--')
        axs_clean[0][col].plot(pareto_runtime_xs, pareto_runtime_ys, color="#ef9b20", linestyle='--')
        axs_clean[1][col].plot(pareto_cost_xs, pareto_cost_ys, color="#ef9b20", linestyle='--')
        axs_clean_bw[0][col].plot(pareto_runtime_xs, pareto_runtime_ys, color="#ef9b20", linestyle='--')
        axs_clean_bw[1][col].plot(pareto_cost_xs, pareto_cost_ys, color="#ef9b20", linestyle='--')
        axs_clean_mc[0][col].plot(pareto_runtime_xs, pareto_runtime_ys, color="#ef9b20", linestyle='--')
        axs_clean_mc[1][col].plot(pareto_cost_xs, pareto_cost_ys, color="#ef9b20", linestyle='--')

        # set x,y-lim for each workload
        left, right = -0.05, 1.05
        if workload == "real-estate":
            left = 0.65
            right = 0.9
        elif workload == "biofabric":
            left = 0.3
            right = 0.6
        axs_text[0][col].set_xlim(left, right)
        axs_text[0][col].set_ylim(ymin=0)
        axs_text[1][col].set_xlim(left, right)
        if workload != "biofabric":
            axs_text[1][col].set_ylim(ymin=0)
        axs_clean[0][col].set_xlim(left, right)
        axs_clean[0][col].set_ylim(ymin=0)
        axs_clean[1][col].set_xlim(left, right)
        if workload != "biofabric":
            axs_clean[1][col].set_ylim(ymin=0)
        axs_clean_bw[0][col].set_xlim(left, right)
        axs_clean_bw[0][col].set_ylim(ymin=0)
        axs_clean_bw[1][col].set_xlim(left, right)
        if workload != "biofabric":
            axs_clean_bw[1][col].set_ylim(ymin=0)
        axs_clean_mc[0][col].set_xlim(left, right)
        axs_clean_mc[0][col].set_ylim(ymin=0)
        axs_clean_mc[1][col].set_xlim(left, right)
        if workload != "biofabric":
            axs_clean_mc[1][col].set_ylim(ymin=0)

        # turn on grid lines
        axs_text[0][col].grid(True, alpha=0.4)
        axs_text[1][col].grid(True, alpha=0.4)
        axs_clean[0][col].grid(True, alpha=0.4)
        axs_clean[1][col].grid(True, alpha=0.4)
        axs_clean_bw[0][col].grid(True, alpha=0.4)
        axs_clean_bw[1][col].grid(True, alpha=0.4)
        axs_clean_mc[0][col].grid(True, alpha=0.4)
        axs_clean_mc[1][col].grid(True, alpha=0.4)

    # remove ticks from first row
    for idx in range(3):
        axs_text[0][idx].set_xticklabels([])
        axs_clean[0][idx].set_xticklabels([])
        axs_clean_bw[0][idx].set_xticklabels([])
        axs_clean_mc[0][idx].set_xticklabels([])

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
        axs_clean_bw[0][idx].set_title(f"{title}", fontsize=12)
        axs_clean_mc[0][idx].set_title(f"{title}", fontsize=12)

    axs_text[0][0].set_ylabel("Runtime (seconds)", fontsize=12)
    axs_text[1][0].set_ylabel("Cost (USD)", fontsize=12)
    for idx in range(3):
        axs_text[1][idx].set_xlabel("F1 Score", fontsize=12)
    
    axs_clean[0][0].set_ylabel("Runtime (seconds)", fontsize=12)
    axs_clean[1][0].set_ylabel("Cost (USD)", fontsize=12)
    for idx in range(3):
        axs_clean[1][idx].set_xlabel("F1 Score", fontsize=12)

    axs_clean_bw[0][0].set_ylabel("Runtime (seconds)", fontsize=12)
    axs_clean_bw[1][0].set_ylabel("Cost (USD)", fontsize=12)
    for idx in range(3):
        axs_clean_bw[1][idx].set_xlabel("F1 Score", fontsize=12)

    axs_clean_mc[0][0].set_ylabel("Runtime (seconds)", fontsize=12)
    axs_clean_mc[1][0].set_ylabel("Cost (USD)", fontsize=12)
    for idx in range(3):
        axs_clean_mc[1][idx].set_xlabel("F1 Score", fontsize=12)

    fig_text.savefig(f"final-eval-results/plots/all-text.png", dpi=500, bbox_inches="tight")
    fig_clean.savefig(f"final-eval-results/plots/all-clean.png", dpi=500, bbox_inches="tight")
    fig_clean_bw.savefig(f"final-eval-results/plots/all-clean-bw.png", dpi=500, bbox_inches="tight")
    fig_clean_mc.savefig(f"final-eval-results/plots/all-clean-mc.png", dpi=500, bbox_inches="tight")


def plot_reopt(results, policy):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15,6))

    # parse results into fields
    results_df = pd.DataFrame(results)

    plan_to_ord = {"Best": 0, "PZ": 1, "Naive": 2}
    policy_df = results_df[results_df.policy==policy]
    policy_df['ord'] = policy_df.plan.apply(lambda plan: plan_to_ord[plan])
    policy_df.sort_values(by='ord', inplace=True)

    g = sns.barplot(
        data=policy_df, # kind="bar",
        x="workload", y="cost", hue="plan",
        palette=["#87bc45", "#27aeef", "#b33dc6"], alpha=.6, # height=6,
        ax=axs[0], # order=["Best", "PZ", "Naive"],
    )
    g.legend_.set_title(None)
    g.set_xlabel(None)
    g.set_ylabel(None)

    g = sns.barplot(
        data=policy_df, # kind="bar",
        x="workload", y="runtime", hue="plan",
        palette=["#87bc45", "#27aeef", "#b33dc6"], alpha=.6, # height=6,
        ax=axs[1], # order=["Best", "PZ", "Naive"],
    )
    g.legend_.remove()
    g.set_xlabel(None)
    g.set_ylabel(None)

    g = sns.barplot(
        data=policy_df, # kind="bar",
        x="workload", y="f1_score", hue="plan",
        palette=["#87bc45", "#27aeef", "#b33dc6"], alpha=.6, # height=6,
        ax=axs[2], # order=["Best", "PZ", "Naive"],
    )
    g.legend_.remove()
    g.set_xlabel(None)
    g.set_ylabel(None)
    axs[0].set_title("Cost (USD)", fontsize=15)
    axs[1].set_title("Runtime (Seconds)", fontsize=15)
    axs[2].set_title("F1-Score", fontsize=15)
    if policy == "max-quality-at-fixed-cost":
        fig.suptitle("Max Quality @ Fixed Cost")
        axs[0].axhline(y=20.0, xmin=0.0, xmax=0.5, color='#ef9b20', linestyle='--')
        axs[0].axhline(y=3.0, xmin=0.5, xmax=1, color='#ef9b20', linestyle='--')
    elif policy == "max-quality-at-fixed-runtime":
        fig.suptitle("Max Quality @ Fixed Rutnime")
    elif policy == "min-cost-at-fixed-quality":
        fig.suptitle("Min Cost @ Fixed Quality")
    elif policy == "min-runtime-at-fixed-quality":
        fig.suptitle("Min Runtime @ Fixed Quality")
    fig.supxlabel('Workload')
    # fig.supylabel('Percent Error')
    fig.savefig(f"final-eval-results/plots/reopt-{policy}.png", dpi=500, bbox_inches="tight")


def plot_reopt2(results, workload):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15,6))

    # parse results into fields
    results_df = pd.DataFrame(results)

    plan_to_ord = {"Best": 0, "PZ": 1, "Naive": 2}
    # policy_df = results_df[results_df.policy==policy]
    results_df['ord'] = results_df.plan.apply(lambda plan: plan_to_ord[plan])
    results_df.sort_values(by='ord', inplace=True)

    g = sns.barplot(
        data=results_df, # kind="bar",
        x="policy", y="cost", hue="plan",
        palette=["#87bc45", "#b33dc6"], alpha=.6, # height=6,
        ax=axs[0], # order=["Best", "PZ", "Naive"],
    )
    g.legend_.set_title(None)
    g.set_xlabel(None)
    g.set_ylabel(None)

    g = sns.barplot(
        data=results_df, # kind="bar",
        x="policy", y="runtime", hue="plan",
        palette=["#87bc45", "#b33dc6"], alpha=.6, # height=6,
        ax=axs[1], # order=["Best", "PZ", "Naive"],
    )
    g.legend_.remove()
    g.set_xlabel(None)
    g.set_ylabel(None)

    g = sns.barplot(
        data=results_df, # kind="bar",
        x="policy", y="f1_score", hue="plan",
        palette=["#87bc45", "#b33dc6"], alpha=.6, # height=6,
        ax=axs[2], # order=["Best", "PZ", "Naive"],
    )
    g.legend_.remove()
    g.set_xlabel(None)
    g.set_ylabel(None)
    axs[0].tick_params(axis='x', rotation=45)
    axs[1].tick_params(axis='x', rotation=45)
    axs[2].tick_params(axis='x', rotation=45)
    axs[0].set_title("Cost (USD)", fontsize=15)
    axs[1].set_title("Runtime (Seconds)", fontsize=15)
    axs[2].set_title("F1-Score", fontsize=15)
    fig.suptitle(f"Re-Optimization on {workload} by Policy")
    # if policy == "max-quality-at-fixed-cost":
    #     fig.suptitle("Max Quality @ Fixed Cost")
    #     axs[0].axhline(y=20.0, xmin=0.0, xmax=0.5, color='#ef9b20', linestyle='--')
    #     axs[0].axhline(y=3.0, xmin=0.5, xmax=1, color='#ef9b20', linestyle='--')
    # elif policy == "max-quality-at-fixed-runtime":
    #     fig.suptitle("Max Quality @ Fixed Rutnime")
    # elif policy == "min-cost-at-fixed-quality":
    #     fig.suptitle("Min Cost @ Fixed Quality")
    # elif policy == "min-runtime-at-fixed-quality":
    #     fig.suptitle("Min Runtime @ Fixed Quality")
    # fig.supxlabel('Policy')
    # fig.supylabel('Percent Error')
    fig.savefig(f"final-eval-results/plots/reopt2-{workload}.png", dpi=500, bbox_inches="tight")


if __name__ == "__main__":
    # parse arguments
    startTime = time.time()
    parser = argparse.ArgumentParser(description='Run the evaluation(s) for the paper')
    parser.add_argument('--all', default=False, action='store_true', help='')
    parser.add_argument('--reopt', default=False, action='store_true', help='')
    # parser.add_argument('--workload', type=str, help='The workload: one of ["biofabric", "enron", "real-estate"]')
    # parser.add_argument('--opt' , type=str, help='The optimization: one of ["model", "codegen", "token-reduction"]')

    args = parser.parse_args()

    # create directory for intermediate results
    os.makedirs(f"final-eval-results/plots", exist_ok=True)

    if args.all:
        # opt and workload to # of plots
        opt_workload_to_num_plans = {
            "model": {
                "enron": 11,
                "real-estate": 10,
                "biofabric": 14,
            },
            "codegen": {
                "enron": 6,
                "real-estate": 7,
                "biofabric": 6,
            },
            "token-reduction": {
                "enron": 12,
                "real-estate": 16,
                "biofabric": 16,
            },
        }

        # read results file(s) generated by evaluate_pz_plans
        results = {
            "enron": {
                "model": [],
                "codegen": [],
                "token-reduction": [],
            },
            "real-estate": {
                "model": [],
                "codegen": [],
                "token-reduction": [],
            },
            "biofabric": {
                "model": [],
                "codegen": [],
                "token-reduction": [],
            },
        }
        for workload in ["enron", "real-estate", "biofabric"]:
            for opt in ["model", "codegen", "token-reduction"]:
                num_plans = opt_workload_to_num_plans[opt][workload]
                for plan_idx in range(num_plans):
                    if workload == "biofabric" and opt == "codegen" and plan_idx == 3:
                        continue

                    with open(f"final-eval-results/{opt}/{workload}/results-{plan_idx}.json", 'r') as f:
                        result = json.load(f)
                        results[workload][opt].append((plan_idx, result))

        plot_runtime_cost_vs_quality(results)

    if args.reopt:
        policy_to_best_plan = {
            "max-quality-at-fixed-cost": {
                "enron": ("model", 5),
                "real-estate": ("token-reduction", 8),
                "biofabric": ("token-reduction", 11),
            },
            "max-quality-at-fixed-runtime": {
                "enron": ("model", 6),
                "real-estate": ("token-reduction", 8),
                "biofabric": ("token-reduction", 8),
            },
            "min-cost-at-fixed-quality": {
                "enron": ("model", 6),
                "real-estate": ("codegen", 6),
                "biofabric": ("token-reduction", 11),
            },
            "min-runtime-at-fixed-quality": {
                "enron": ("model", 6),
                "real-estate": ("codegen", 6),
                "biofabric": ("model", 3),
            },
        }
        policy_to_naive_plan = {
            ### max quality
            "max-quality-at-fixed-cost": {
                "enron": ("token-reduction", 1),
                "real-estate": ("token-reduction", 9),
                "biofabric": ("token-reduction", 4),
            },
            ### max quality
            "max-quality-at-fixed-runtime": {
                "enron": ("token-reduction", 1),
                "real-estate": ("token-reduction", 9),
                "biofabric": ("token-reduction", 4),
            },
            ### min cost
            "min-cost-at-fixed-quality": {
                "enron": ("token-reduction", 0), # ("model", 10),
                "real-estate": ("codegen", 0),
                "biofabric": ("token-reduction", 8),
            },
            ### min runtime
            "min-runtime-at-fixed-quality": {
                "enron": ("token-reduction", 0), # ("model", 10),
                "real-estate": ("codegen", 0),
                "biofabric": ("model", 0),
            },
        }
        for workload in ["enron", "real-estate"]:
            results = []
            for policy in ["max-quality-at-fixed-cost", "max-quality-at-fixed-runtime", "min-cost-at-fixed-quality", "min-runtime-at-fixed-quality"]:
            # for workload in ["enron", "real-estate", "biofabric"]:
            
                with open(f'final-eval-results/reoptimization/{workload}/{policy}.json', 'r') as f:
                    result_dict = json.load(f)
                    results.append({"plan": "PZ", "policy": policy, "workload": workload, "f1_score": result_dict["f1_score"], "cost": result_dict["cost"], "runtime": result_dict["runtime"]})

                # best_plan_opt, best_plan_idx = policy_to_best_plan[policy][workload]
                # with open(f'final-eval-results/{best_plan_opt}/{workload}/results-{best_plan_idx}.json', 'r') as f:
                #     result_dict = json.load(f)
                #     results.append({"plan": "Best", "policy": policy, "workload": workload, "f1_score": result_dict["f1_score"], "cost": result_dict["cost"], "runtime": result_dict["runtime"]})
                
                naive_plan_opt, naive_plan_idx = policy_to_naive_plan[policy][workload]
                with open(f'final-eval-results/{naive_plan_opt}/{workload}/results-{naive_plan_idx}.json', 'r') as f:
                    result_dict = json.load(f)
                    results.append({"plan": "Naive", "policy": policy, "workload": workload, "f1_score": result_dict["f1_score"], "cost": result_dict["cost"], "runtime": result_dict["runtime"]})

            # plot_reopt(results, policy)
            plot_reopt2(results, workload)
