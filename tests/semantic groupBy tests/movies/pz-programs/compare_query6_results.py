#!/usr/bin/env python3
"""
Compare Query 6 results from PZ with ground truth.
Generates a styled summary table image similar to the analysis summary_table.png.
"""

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


# ─── Styling (matches analyze.py) ─────────────────────────────────────────────

HEADER_COLOR = "#1E3A5F"
ROW_ALT_COLOR = "#F7F9FC"
ROW_LABEL_COLOR = "#E8EDF5"
EDGE_COLOR = "#CCCCCC"


def style_table(tbl, n_data_rows: int):
    """Apply the shared header/row styling."""
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor(HEADER_COLOR)
            cell.set_text_props(color="white", fontweight="bold")
        elif c == -1:
            cell.set_facecolor(ROW_LABEL_COLOR)
            cell.set_text_props(fontweight="bold")
        else:
            cell.set_facecolor(ROW_ALT_COLOR if r % 2 == 0 else "white")
        cell.set_edgecolor(EDGE_COLOR)


def make_stats_subtable(ax, comparison: pd.DataFrame):
    """Left sub-table: summary statistics."""
    ax.axis("off")

    exact = (comparison["Difference"] < 1e-9).sum()
    close = (comparison["Difference"] <= 0.1).sum()
    n = len(comparison)

    rows = [
        ["Directors compared",  str(n)],
        ["Exact matches",        f"{exact}  ({100*exact/n:.1f}%)"],
        ["Within ±0.1",          f"{close}  ({100*close/n:.1f}%)"],
        ["Mean |difference|",    f"{comparison['Difference'].mean():.4f}"],
        ["Std deviation",        f"{comparison['Difference'].std():.4f}"],
        ["Min difference",       f"{comparison['Difference'].min():.4f}"],
        ["Max difference",       f"{comparison['Difference'].max():.4f}"],
    ]

    tbl = ax.table(
        cellText=[[r[1]] for r in rows],
        rowLabels=[r[0] for r in rows],
        colLabels=["Value"],
        cellLoc="center",
        loc="center",
    )
    style_table(tbl, len(rows))
    # ax.set_title("Summary Statistics", fontsize=11, fontweight="bold", pad=10)


def make_score_subtable(ax, comparison: pd.DataFrame):
    """Middle sub-table: distribution of differences by bucket."""
    ax.axis("off")

    diff = comparison["Difference"]
    buckets = [
        ("= 0.0",      diff < 1e-9),
        ("0.0 – 0.1",  (diff >= 1e-9) & (diff <= 0.1)),
        ("0.1 – 0.2",  (diff > 0.1)   & (diff <= 0.2)),
        ("0.2 – 0.3",  (diff > 0.2)   & (diff <= 0.3)),
        ("> 0.3",      diff > 0.3),
    ]
    n = len(comparison)
    cell_data = [[str(mask.sum()), f"{100*mask.sum()/n:.1f}%"] for _, mask in buckets]
    row_labels = [label for label, _ in buckets]

    tbl = ax.table(
        cellText=cell_data,
        rowLabels=row_labels,
        colLabels=["Count", "% of Total"],
        cellLoc="center",
        loc="center",
    )
    style_table(tbl, len(buckets))
    # ax.set_title("Difference Distribution", fontsize=11, fontweight="bold", pad=10)


def make_all_directors_subtable(ax, comparison: pd.DataFrame):
    """Full directors table showing all rows."""
    ax.axis("off")

    cell_data = [
        [row["Director"], f"{row['PZ Score']:.3f}", f"{row['Ground Truth Score']:.3f}", f"{row['Difference']:.3f}"]
        for _, row in comparison.iterrows()
    ]

    tbl = ax.table(
        cellText=cell_data,
        colLabels=["Director", "PZ", "GT", "Diff"],
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.4)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor(HEADER_COLOR)
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor(ROW_ALT_COLOR if r % 2 == 0 else "white")
        cell.set_edgecolor(EDGE_COLOR)

    # ax.set_title("All Directors", fontsize=11, fontweight="bold", pad=10)


def save_summary_figure(comparison_table: pd.DataFrame, output_path: Path):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    n = len(comparison_table)
    # Top section: stats + distribution side by side
    # Bottom section: full directors table spanning full width
    fig = plt.figure(figsize=(14, 4.5 + n * 0.28))
    fig.suptitle(
        "Query 6 — PZ vs Ground Truth",
        fontsize=13, fontweight="bold",
    )

    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, n * 0.28 / 4.5], width_ratios=[1, 1])

    ax_stats = fig.add_subplot(gs[0, 0])
    ax_dist  = fig.add_subplot(gs[0, 1])
    ax_all   = fig.add_subplot(gs[1, :])

    make_stats_subtable(ax_stats, comparison_table)
    make_score_subtable(ax_dist, comparison_table)
    make_all_directors_subtable(ax_all, comparison_table)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Summary figure saved to: {output_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    script_dir = Path(__file__).parent

    pz_results = pd.read_csv(script_dir / "query6_pz_output.csv")
    ground_truth = pd.read_csv(script_dir / "../queries/query6_ground_truth.csv")

    comparison = pz_results.merge(
        ground_truth[["director", "normalizedScore"]],
        on="director",
        how="inner",
        suffixes=("_pz", "_gt"),
    )
    comparison["difference"] = abs(comparison["normalizedScore_pz"] - comparison["normalizedScore_gt"])

    comparison_table = comparison[["director", "normalizedScore_pz", "normalizedScore_gt", "difference"]].copy()
    comparison_table.columns = ["Director", "PZ Score", "Ground Truth Score", "Difference"]

    avg_difference = comparison_table["Difference"].mean()
    variance_difference = comparison_table["Difference"].var()
    std_difference = comparison_table["Difference"].std()

    print("\n" + "="*80)
    print("QUERY 6 COMPARISON: PZ vs Ground Truth")
    print("="*80 + "\n")
    display_table = comparison_table.copy()
    display_table["PZ Score"] = display_table["PZ Score"].map(lambda x: f"{x:.3f}")
    display_table["Ground Truth Score"] = display_table["Ground Truth Score"].map(lambda x: f"{x:.3f}")
    display_table["Difference"] = display_table["Difference"].map(lambda x: f"{x:.3f}")
    print(display_table.to_string(index=False))
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    print(f"Number of directors compared: {len(comparison_table)}")
    print(f"Average difference:           {avg_difference:.4f}")
    print(f"Variance of difference:       {variance_difference:.4f}")
    print(f"Standard deviation:           {std_difference:.4f}")
    print(f"Min difference:               {comparison_table['Difference'].min():.4f}")
    print(f"Max difference:               {comparison_table['Difference'].max():.4f}")
    print("="*80 + "\n")

    output_file = script_dir / "query6_comparison.csv"
    comparison_table.to_csv(output_file, index=False)

    stats_file = script_dir / "query6_comparison_stats.txt"
    with open(stats_file, "w") as f:
        f.write("QUERY 6 COMPARISON STATISTICS\n")
        f.write("="*50 + "\n")
        f.write(f"Number of directors compared: {len(comparison_table)}\n")
        f.write(f"Average difference:           {avg_difference:.4f}\n")
        f.write(f"Variance of difference:       {variance_difference:.4f}\n")
        f.write(f"Standard deviation:           {std_difference:.4f}\n")
        f.write(f"Min difference:               {comparison_table['Difference'].min():.4f}\n")
        f.write(f"Max difference:               {comparison_table['Difference'].max():.4f}\n")

    print(f"Comparison table saved to: {output_file}")
    print(f"Statistics saved to:       {stats_file}")

    figure_file = script_dir / "query6_summary_table.png"
    save_summary_figure(comparison_table, figure_file)


if __name__ == "__main__":
    main()
