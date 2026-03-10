#!/usr/bin/env python3
"""
Run baseline group-by tests: execute baseline implementations (sem_map + groupby),
compare outputs against ground truth, and log performance metrics.

Results are written to the same results/ folder as the sem_groupby tests,
with '_baseline' suffixed filenames so both approaches can be compared side-by-side.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype


BASE_DIR = Path(__file__).resolve().parent
QUERIES_DIR = BASE_DIR / "queries"
BASELINE_DIR = BASE_DIR / "pz-baseline"
RESULTS_DIR = BASE_DIR / "results"


def _discover_tests() -> list[dict[str, Path]]:
    """Find matching pairs of ground-truth query scripts and baseline scripts."""
    query_files = {}
    for path in QUERIES_DIR.glob("query_*.py"):
        parts = path.stem.split("_")
        if len(parts) == 2 and parts[1].isdigit():
            query_files[int(parts[1])] = path

    baseline_files = {}
    for path in BASELINE_DIR.glob("query_*_baseline.py"):
        parts = path.stem.split("_")
        if len(parts) == 3 and parts[1].isdigit() and parts[2] == "baseline":
            baseline_files[int(parts[1])] = path

    test_ids = sorted(set(query_files).intersection(baseline_files))
    tests = []
    for test_id in test_ids:
        tests.append({
            "id": test_id,
            "query_script": query_files[test_id],
            "baseline_script": baseline_files[test_id],
        })
    return tests


def _run_script(script_path: Path, cwd: Path, args: list[str]) -> None:
    cmd = [sys.executable, str(script_path), *args]
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _ground_truth_output_path(test_id: int) -> Path:
    return QUERIES_DIR / f"query{test_id}_ground_truth.csv"


def _compare_outputs(gt_df: pd.DataFrame, baseline_df: pd.DataFrame, tol: float) -> dict[str, Any]:
    common_cols = sorted(set(gt_df.columns).intersection(baseline_df.columns))
    if not common_cols:
        return {
            "pass": False,
            "reason": "no_common_columns",
            "num_rows_gt": len(gt_df),
            "num_rows_baseline": len(baseline_df),
        }

    key_cols = []
    numeric_cols = []
    for col in common_cols:
        gt_is_num = is_numeric_dtype(gt_df[col])
        bl_is_num = is_numeric_dtype(baseline_df[col])
        if gt_is_num and bl_is_num:
            numeric_cols.append(col)
        else:
            key_cols.append(col)

    if key_cols:
        merged = gt_df.merge(
            baseline_df,
            on=key_cols,
            how="outer",
            suffixes=("_gt", "_baseline"),
            indicator=True,
        )
        missing_in_baseline = (merged["_merge"] == "left_only").sum()
        missing_in_gt = (merged["_merge"] == "right_only").sum()
        compare_rows = merged[merged["_merge"] == "both"]
    else:
        gt_sorted = gt_df.sort_values(by=common_cols).reset_index(drop=True)
        bl_sorted = baseline_df.sort_values(by=common_cols).reset_index(drop=True)
        min_len = min(len(gt_sorted), len(bl_sorted))
        compare_rows = pd.concat(
            [
                gt_sorted.iloc[:min_len].add_suffix("_gt"),
                bl_sorted.iloc[:min_len].add_suffix("_baseline"),
            ],
            axis=1,
        )
        missing_in_baseline = max(0, len(gt_sorted) - len(bl_sorted))
        missing_in_gt = max(0, len(bl_sorted) - len(gt_sorted))

    metrics: dict[str, Any] = {
        "num_rows_gt": len(gt_df),
        "num_rows_baseline": len(baseline_df),
        "missing_in_baseline": int(missing_in_baseline),
        "missing_in_gt": int(missing_in_gt),
        "num_compared": int(len(compare_rows)),
    }

    max_abs_error = None
    mean_abs_error = None
    mismatched_rows = 0

    if numeric_cols:
        abs_errors = []
        norm_errors = []
        for col in numeric_cols:
            gt_col = f"{col}_gt"
            bl_col = f"{col}_baseline"
            if gt_col not in compare_rows or bl_col not in compare_rows:
                continue
            diff = (compare_rows[gt_col] - compare_rows[bl_col]).abs()
            abs_errors.append(diff)
            # Normalize by GT column mean so different-scale metrics contribute equally
            gt_mean = compare_rows[gt_col].abs().mean()
            norm_diff = diff / gt_mean if gt_mean > 0 else diff
            norm_errors.append(norm_diff)

        if abs_errors:
            all_errors = pd.concat(abs_errors, axis=1)
            all_norm_errors = pd.concat(norm_errors, axis=1)
            max_abs_error = float(all_errors.max().max())
            mean_abs_error = float(all_errors.mean().mean())
            mean_norm_error = float(all_norm_errors.mean().mean())
            mismatched_rows = int((all_errors.max(axis=1) > tol).sum())

    metrics.update({
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "mismatched_rows": mismatched_rows,
    })

    passed = (
        missing_in_baseline == 0
        and missing_in_gt == 0
        and (max_abs_error is None or max_abs_error <= tol)
        and mismatched_rows == 0
    )

    metrics["pass"] = bool(passed)
    if mean_abs_error is not None:
        # Normalized MAE: each column's errors are scaled by its GT mean,
        # so large-magnitude metrics (e.g. review_count) don't drown out
        # small-magnitude ones (e.g. frac_positive).
        metrics["quality_score"] = float(max(0.0, 1.0 - mean_norm_error))

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline group-by tests")
    parser.add_argument("--policies", type=str, default="maxquality,mincost,mintime",
                        help="Comma-separated list of policies to run")
    parser.add_argument("--execution-strategy", type=str, default="sequential",
                        help="One of 'sequential', 'pipelined', 'parallel'")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--regen-ground-truth", action="store_true",
                        help="Re-run ground truth scripts even if output already exists")
    parser.add_argument("--ids", type=str, default="",
                        help="Comma-separated test ids to run (e.g., '1,2,3')")
    args = parser.parse_args()

    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    requested_ids = {int(x) for x in args.ids.split(",") if x.strip().isdigit()}

    tests = _discover_tests()
    if requested_ids:
        tests = [t for t in tests if t["id"] in requested_ids]

    if not tests:
        print("No baseline tests found.")
        return

    print(f"Found {len(tests)} test(s): {[t['id'] for t in tests]}")
    print(f"Policies: {policies}")
    print(f"Execution strategy: {args.execution_strategy}\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for test in tests:
        test_id = test["id"]

        # Generate / load ground truth
        gt_output = _ground_truth_output_path(test_id)
        if args.regen_ground_truth or not gt_output.exists():
            print(f"[query {test_id}] Generating ground truth...")
            _run_script(test["query_script"], QUERIES_DIR, [])

        if not gt_output.exists():
            print(f"[query {test_id}] Ground truth missing: {gt_output} — skipping")
            continue

        gt_df = pd.read_csv(gt_output)

        for policy in policies:
            policy_dir = RESULTS_DIR / policy
            policy_dir.mkdir(parents=True, exist_ok=True)

            baseline_output = policy_dir / f"query{test_id}_baseline_output.csv"
            stats_output = policy_dir / f"query{test_id}_baseline_stats.json"

            print(f"[query {test_id}][{policy}] Running baseline...")
            _run_script(
                test["baseline_script"],
                BASELINE_DIR,
                [
                    "--policy", policy,
                    "--execution-strategy", args.execution_strategy,
                    "--output", str(baseline_output),
                    "--stats-output", str(stats_output),
                ],
            )

            baseline_df = pd.read_csv(baseline_output) if baseline_output.exists() else pd.DataFrame()
            compare_metrics = _compare_outputs(gt_df, baseline_df, args.tolerance)

            exec_metrics: dict[str, Any] = {}
            if stats_output.exists():
                with open(stats_output) as f:
                    stats = json.load(f)
                exec_metrics = {
                    "total_execution_time": stats.get("total_execution_time"),
                    "total_execution_cost": stats.get("total_execution_cost"),
                    "total_tokens": stats.get("total_tokens"),
                    "optimization_time": stats.get("optimization_time"),
                    "plan_execution_time": stats.get("plan_execution_time"),
                }

            row = {
                "test_id": test_id,
                "policy": policy,
                **exec_metrics,
                **compare_metrics,
            }
            summary_rows.append(row)

            result_json = policy_dir / f"query{test_id}_baseline_comparison.json"
            with open(result_json, "w") as f:
                json.dump(row, f, indent=2)

            status = "PASS" if compare_metrics.get("pass") else "FAIL"
            print(f"[query {test_id}][{policy}] {status}")

    summary_path = RESULTS_DIR / "baseline_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"\nSummary written to: {summary_path}")


if __name__ == "__main__":
    main()
