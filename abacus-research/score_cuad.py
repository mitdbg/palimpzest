import json
import os

import numpy as np


def compute_final_metrics(metric: str, dir: str, exp_base_name: str):
    qualities = []
    opt_costs, run_costs = [], []
    opt_times, run_times = [], []
    total_costs, total_times = [], []
    print(f"--- {metric} ---")
    for seed in range(10):
        exp_name = f"{exp_base_name}-seed{seed}"
        if os.path.exists(f"{dir}/{exp_name}-metrics.json") is False:
            print(f"Missing {dir}/{exp_name}-metrics.json")
            continue
        with open(f"{dir}/{exp_name}-metrics.json") as f:
            metrics = json.load(f)
        qualities.append(metrics["f1"])
        opt_costs.append(metrics["optimization_cost"])
        opt_times.append(metrics["optimization_time"])
        run_costs.append(metrics["plan_execution_cost"])
        run_times.append(metrics["plan_execution_time"])
        total_costs.append(metrics["total_execution_cost"])
        total_times.append(metrics["total_execution_time"])
    
    print(f"Opt. Cost: {np.mean(opt_costs):.3f} +/- {np.std(opt_costs):.3f}")
    print(f"Opt. Time: {np.mean(opt_times):.3f} +/- {np.std(opt_times):.3f}")
    print(f"Run Cost: {np.mean(run_costs):.3f} +/- {np.std(run_costs):.3f}")
    print(f"Run Time: {np.mean(run_times):.3f} +/- {np.std(run_times):.3f}")
    print(f"Total Cost: {np.mean(total_costs):.3f} +/- {np.std(total_costs):.3f}")
    print(f"Total Time: {np.mean(total_times):.3f} +/- {np.std(total_times):.3f}")
    print(f"Quality: {np.mean(qualities):.3f} +/- {np.std(qualities):.3f}")
    print("-------")

if __name__ == "__main__":
    compute_final_metrics("quality", "opt-profiling-data", "cuad-final-mab-k6-j4-budget50")
    compute_final_metrics("cost", "opt-profiling-data", "cuad-final-mincost-k6-j4-budget50")
    compute_final_metrics("latency", "opt-profiling-data", "cuad-final-minlatency-k6-j4-budget50")