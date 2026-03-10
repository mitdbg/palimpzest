#!/usr/bin/env python3
"""
Run semantic group-by tests: generate ground truth, execute PZ programs,
compare outputs, and log performance metrics.
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
PZ_DIR = BASE_DIR / "pz-programs"
RESULTS_DIR = BASE_DIR / "results"


def _discover_tests() -> list[dict[str, Path]]:
    query_files = {}
    for path in QUERIES_DIR.glob("query_*.py"):
        parts = path.stem.split("_")
        if len(parts) == 2 and parts[1].isdigit():
            query_files[int(parts[1])] = path

    pz_files = {}
    for path in PZ_DIR.glob("query_*_pz.py"):
        parts = path.stem.split("_")
        if len(parts) == 3 and parts[1].isdigit() and parts[2] == "pz":
            pz_files[int(parts[1])] = path

    test_ids = sorted(set(query_files).intersection(pz_files))
    tests = []
    for test_id in test_ids:
        tests.append({
            "id": test_id,
            "query_script": query_files[test_id],
            "pz_script": pz_files[test_id],
        })
    return tests


def _run_script(script_path: Path, cwd: Path, args: list[str]) -> None:
    cmd = [sys.executable, str(script_path), *args]
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _ground_truth_output_path(test_id: int) -> Path:
    return QUERIES_DIR / f"query{test_id}_ground_truth.csv"


def _compare_outputs(gt_df: pd.DataFrame, pz_df: pd.DataFrame, tol: float) -> dict[str, Any]:
    common_cols = sorted(set(gt_df.columns).intersection(pz_df.columns))
    if not common_cols:
        return {
            "pass": False,
            "reason": "no_common_columns",
            "num_rows_gt": len(gt_df),
            "num_rows_pz": len(pz_df),
        }

    key_cols = []
    numeric_cols = []
    for col in common_cols:
        gt_is_num = is_numeric_dtype(gt_df[col])
        pz_is_num = is_numeric_dtype(pz_df[col])
        if gt_is_num and pz_is_num:
            numeric_cols.append(col)
        else:
            key_cols.append(col)

    if key_cols:
        merged = gt_df.merge(
            pz_df,
            on=key_cols,
            how="outer",
            suffixes=("_gt", "_pz"),
            indicator=True,
        )
        missing_in_pz = (merged["_merge"] == "left_only").sum()
        missing_in_gt = (merged["_merge"] == "right_only").sum()
        compare_rows = merged[merged["_merge"] == "both"]
    else:
        gt_sorted = gt_df.sort_values(by=common_cols).reset_index(drop=True)
        pz_sorted = pz_df.sort_values(by=common_cols).reset_index(drop=True)
        min_len = min(len(gt_sorted), len(pz_sorted))
        compare_rows = pd.concat(
            [
                gt_sorted.iloc[:min_len].add_suffix("_gt"),
                pz_sorted.iloc[:min_len].add_suffix("_pz"),
            ],
            axis=1,
        )
        missing_in_pz = max(0, len(gt_sorted) - len(pz_sorted))
        missing_in_gt = max(0, len(pz_sorted) - len(gt_sorted))

    metrics: dict[str, Any] = {
        "num_rows_gt": len(gt_df),
        "num_rows_pz": len(pz_df),
        "missing_in_pz": int(missing_in_pz),
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
            pz_col = f"{col}_pz"
            if gt_col not in compare_rows or pz_col not in compare_rows:
                continue
            diff = (compare_rows[gt_col] - compare_rows[pz_col]).abs()
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
        missing_in_pz == 0
        and missing_in_gt == 0
        and (max_abs_error is None or max_abs_error <= tol)
        and mismatched_rows == 0
    )

    metrics["pass"] = bool(passed)
    if mean_abs_error is not None:
        # Normalized MAE: each column's errors are scaled by its GT mean,
        # so large-magnitude metrics (e.g. review_count) don't drown out
        # small-magnitude ones (e.g. frac_positive).
        quality_score = max(0.0, 1.0 - mean_norm_error)
        metrics["quality_score"] = float(quality_score)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run semantic group-by tests")
    parser.add_argument("--policies", type=str, default="maxquality,mincost,mintime",
                        help="Comma-separated list of policies to run")
    parser.add_argument("--execution-strategy", type=str, default="sequential",
                        help="One of 'sequential', 'pipelined', 'parallel'")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--regen-ground-truth", action="store_true")
    parser.add_argument("--ids", type=str, default="",
                        help="Comma-separated test ids to run (e.g., '1,2,3')")
    args = parser.parse_args()

    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    requested_ids = {int(x) for x in args.ids.split(",") if x.strip().isdigit()}

    tests = _discover_tests()
    if requested_ids:
        tests = [t for t in tests if t["id"] in requested_ids]

    if not tests:
        print("No tests found.")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for test in tests:
        test_id = test["id"]
        gt_output = _ground_truth_output_path(test_id)
        if args.regen_ground_truth or not gt_output.exists():
            _run_script(test["query_script"], QUERIES_DIR, [])

        if not gt_output.exists():
            print(f"Ground truth missing for query {test_id}: {gt_output}")
            continue

        gt_df = pd.read_csv(gt_output)

        for policy in policies:
            policy_dir = RESULTS_DIR / policy
            policy_dir.mkdir(parents=True, exist_ok=True)
            pz_output = policy_dir / f"query{test_id}_pz_output.csv"
            stats_output = policy_dir / f"query{test_id}_pz_stats.json"

            _run_script(
                test["pz_script"],
                PZ_DIR,
                [
                    "--policy", policy,
                    "--execution-strategy", args.execution_strategy,
                    "--output", str(pz_output),
                    "--stats-output", str(stats_output),
                ],
            )

            pz_df = pd.read_csv(pz_output) if pz_output.exists() else pd.DataFrame()
            compare_metrics = _compare_outputs(gt_df, pz_df, args.tolerance)

            exec_metrics: dict[str, Any] = {}
            if stats_output.exists():
                with open(stats_output, "r") as f:
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

            result_json = policy_dir / f"query{test_id}_comparison.json"
            with open(result_json, "w") as f:
                json.dump(row, f, indent=2)

            status = "PASS" if compare_metrics.get("pass") else "FAIL"
            print(f"[query {test_id}][{policy}] {status}")

    summary_path = RESULTS_DIR / "summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"\nSummary written to: {summary_path}")


if __name__ == "__main__":
    main()
