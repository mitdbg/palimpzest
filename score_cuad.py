import json

import numpy as np

if __name__ == "__main__":
    qualities = []
    opt_costs, run_costs = [], []
    opt_times, run_times = [], []
    total_costs, total_times = [], []
    for seed in range(10):
        exp_name = f"cuad-final-mab-k6-j4-budget50-seed{seed}"
        with open(f"opt-profiling-data/{exp_name}-metrics.json") as f:
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
