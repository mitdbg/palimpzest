#!/usr/bin/env python3
"""
Rerun comparisons against already-generated PZ / baseline outputs.

What this script does:
  1. Regenerates the ground-truth CSVs for Q3 and Q5 (fixing input inconsistencies).
  2. Recomputes comparison metrics (using normalized MAE quality) for every query
     and system (pz / baseline) using the *existing* output CSVs — the PZ and
     baseline programs themselves are NOT re-run.
  3. Writes updated comparison JSONs.
  4. Calls analyze.py to regenerate all figures and tables.

Usage:
    python rerun_comparison.py [--policies maxquality] [--ids 3,5]
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

BASE_DIR   = Path(__file__).resolve().parent
QUERIES_DIR = BASE_DIR / "queries"
RESULTS_DIR = BASE_DIR / "results"
ANALYZE_SCRIPT = RESULTS_DIR / "analyze.py"


# ─── Quality metric (normalized MAE) ──────────────────────────────────────────

def _compare_outputs(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    pred_suffix: str,   # "pz" or "baseline"
    tol: float,
) -> dict[str, Any]:
    common_cols = sorted(set(gt_df.columns).intersection(pred_df.columns))
    if not common_cols:
        return {
            "pass": False,
            "reason": "no_common_columns",
            "num_rows_gt": len(gt_df),
            f"num_rows_{pred_suffix}": len(pred_df),
        }

    key_cols, numeric_cols = [], []
    for col in common_cols:
        if is_numeric_dtype(gt_df[col]) and is_numeric_dtype(pred_df[col]):
            numeric_cols.append(col)
        else:
            key_cols.append(col)

    if key_cols:
        merged = gt_df.merge(
            pred_df,
            on=key_cols,
            how="outer",
            suffixes=("_gt", f"_{pred_suffix}"),
            indicator=True,
        )
        missing_in_pred = int((merged["_merge"] == "left_only").sum())
        missing_in_gt   = int((merged["_merge"] == "right_only").sum())
        compare_rows = merged[merged["_merge"] == "both"]
    else:
        gt_s   = gt_df.sort_values(by=common_cols).reset_index(drop=True)
        pred_s = pred_df.sort_values(by=common_cols).reset_index(drop=True)
        n = min(len(gt_s), len(pred_s))
        compare_rows = pd.concat(
            [gt_s.iloc[:n].add_suffix("_gt"), pred_s.iloc[:n].add_suffix(f"_{pred_suffix}")],
            axis=1,
        )
        missing_in_pred = max(0, len(gt_s) - len(pred_s))
        missing_in_gt   = max(0, len(pred_s) - len(gt_s))

    metrics: dict[str, Any] = {
        "num_rows_gt": len(gt_df),
        f"num_rows_{pred_suffix}": len(pred_df),
        f"missing_in_{pred_suffix}": missing_in_pred,
        "missing_in_gt": missing_in_gt,
        "num_compared": len(compare_rows),
    }

    max_abs_error = mean_abs_error = mean_norm_error = None
    mismatched_rows = 0

    if numeric_cols and len(compare_rows) > 0:
        abs_errors, norm_errors = [], []
        for col in numeric_cols:
            gt_col   = f"{col}_gt"
            pred_col = f"{col}_{pred_suffix}"
            if gt_col not in compare_rows or pred_col not in compare_rows:
                continue
            diff = (compare_rows[gt_col] - compare_rows[pred_col]).abs()
            abs_errors.append(diff)
            gt_mean = compare_rows[gt_col].abs().mean()
            norm_errors.append(diff / gt_mean if gt_mean > 0 else diff)

        if abs_errors:
            all_abs  = pd.concat(abs_errors,  axis=1)
            all_norm = pd.concat(norm_errors, axis=1)
            max_abs_error  = float(all_abs.max().max())
            mean_abs_error = float(all_abs.mean().mean())
            mean_norm_error = float(all_norm.mean().mean())
            mismatched_rows = int((all_abs.max(axis=1) > tol).sum())

    metrics.update({
        "max_abs_error":    max_abs_error,
        "mean_abs_error":   mean_abs_error,
        "mismatched_rows":  mismatched_rows,
    })

    passed = (
        missing_in_pred == 0
        and missing_in_gt == 0
        and (max_abs_error is None or max_abs_error <= tol)
        and mismatched_rows == 0
    )
    metrics["pass"] = bool(passed)

    if mean_norm_error is not None:
        metrics["quality_score"] = float(max(0.0, 1.0 - mean_norm_error))

    return metrics


# ─── Ground-truth regeneration ────────────────────────────────────────────────

def _regen_ground_truth(query_ids: list[int]) -> None:
    """Re-run the GT scripts for the given query IDs."""
    for qid in query_ids:
        script = QUERIES_DIR / f"query_{qid}.py"
        if not script.exists():
            print(f"  [GT] query_{qid}.py not found — skipping")
            continue
        print(f"  [GT] Regenerating ground truth for query {qid}...")
        subprocess.run(
            [sys.executable, str(script)],
            cwd=str(QUERIES_DIR),
            check=True,
        )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Rerun comparison (no PZ re-execution)")
    parser.add_argument("--policies", default="maxquality",
                        help="Comma-separated policies (default: maxquality)")
    parser.add_argument("--ids", default="",
                        help="Comma-separated query IDs to update (default: all found)")
    parser.add_argument("--regen-gt-ids", default="5",
                        help="Comma-separated IDs whose GT CSV should be regenerated "
                             "(default: 5, which had the approxTone→emotionalTone fix)")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--skip-analyze", action="store_true",
                        help="Skip calling analyze.py at the end")
    args = parser.parse_args()

    policies      = [p.strip() for p in args.policies.split(",") if p.strip()]
    regen_gt_ids  = [int(x) for x in args.regen_gt_ids.split(",") if x.strip().isdigit()]
    requested_ids = {int(x) for x in args.ids.split(",") if x.strip().isdigit()}

    # Step 1 – regenerate ground-truth CSVs for the fixed queries
    if regen_gt_ids:
        print("\n── Regenerating ground-truth CSVs ──")
        _regen_ground_truth(regen_gt_ids)

    # Step 2 – find all queries that have a ground-truth CSV
    gt_paths = {
        int(p.stem.replace("query", "").replace("_ground_truth", "")): p
        for p in QUERIES_DIR.glob("query*_ground_truth.csv")
    }
    if requested_ids:
        gt_paths = {k: v for k, v in gt_paths.items() if k in requested_ids}

    print(f"\n── Recomputing comparisons for queries: {sorted(gt_paths)} ──")

    for policy in policies:
        policy_dir = RESULTS_DIR / policy
        if not policy_dir.exists():
            print(f"  Policy dir not found: {policy_dir} — skipping")
            continue

        for qid, gt_path in sorted(gt_paths.items()):
            gt_df = pd.read_csv(gt_path)

            for pred_suffix, json_name, csv_name in [
                ("pz",       f"query{qid}_comparison.json",          f"query{qid}_pz_output.csv"),
                ("baseline", f"query{qid}_baseline_comparison.json", f"query{qid}_baseline_output.csv"),
            ]:
                pred_csv  = policy_dir / csv_name
                json_path = policy_dir / json_name

                if not pred_csv.exists():
                    print(f"  [Q{qid}][{policy}][{pred_suffix}] output CSV missing — skipping")
                    continue

                pred_df = pd.read_csv(pred_csv)
                compare = _compare_outputs(gt_df, pred_df, pred_suffix, args.tolerance)

                # Preserve execution stats from the existing JSON
                exec_stats: dict[str, Any] = {}
                if json_path.exists():
                    with open(json_path) as f:
                        old = json.load(f)
                    for key in ("total_execution_time", "total_execution_cost",
                                "total_tokens", "optimization_time", "plan_execution_time"):
                        exec_stats[key] = old.get(key)

                row = {
                    "test_id": qid,
                    "policy":  policy,
                    **exec_stats,
                    **compare,
                }
                with open(json_path, "w") as f:
                    json.dump(row, f, indent=2)

                q_score = compare.get("quality_score", "n/a")
                status  = "PASS" if compare.get("pass") else "FAIL"
                print(f"  [Q{qid}][{policy}][{pred_suffix}] {status}  quality={q_score:.4f}" if isinstance(q_score, float) else f"  [Q{qid}][{policy}][{pred_suffix}] {status}  quality={q_score}")

    # Step 3 – regenerate figures
    if not args.skip_analyze:
        if ANALYZE_SCRIPT.exists():
            print(f"\n── Regenerating figures ({ANALYZE_SCRIPT.name}) ──")
            subprocess.run([sys.executable, str(ANALYZE_SCRIPT)], check=True)
        else:
            print(f"\nanalyze.py not found at {ANALYZE_SCRIPT} — skipping figure generation")

    print("\nDone.")


if __name__ == "__main__":
    main()
