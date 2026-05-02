#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "report" / "figures" / "accuracy_latency_tradeoff.pdf"
NUM_QUERIES = 10895


def _wall_s(ms_per_query: float) -> float:
    return ms_per_query * NUM_QUERIES / 1000.0


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_runs() -> tuple[dict, dict]:
    xml_path = REPO_ROOT / "outputs" / "tvr" / "xml_hierarchical" / "dense_shortlist_xml_eval_summary.json"
    hybrid_path = REPO_ROOT / "outputs" / "tvr" / "xml_hybrid" / "hybrid_eval_summary.json"
    if xml_path.exists() and hybrid_path.exists():
        xml_runs = _load_json(xml_path)["runs"]
        hybrid_runs = {row["method"]: row for row in _load_json(hybrid_path)["runs"]}
        return xml_runs, hybrid_runs

    # Fallback mirrors report/tables/main_results.tex so the figure can be
    # regenerated even when VM-produced output summaries are not in this checkout.
    xml_runs = {
        "full": {"wall_s": _wall_s(7.36), "vcmr_0.5_r1": 0.38},
        "top1": {"wall_s": _wall_s(4.88), "vcmr_0.5_r1": 1.57},
        "top5": {"wall_s": _wall_s(4.96), "vcmr_0.5_r1": 1.43},
        "top10": {"wall_s": _wall_s(5.03), "vcmr_0.5_r1": 1.45},
    }
    hybrid_runs = {
        "shared_compact_to_xml_keep06": {
            "wall_s": _wall_s(11.11),
            "vcmr_r1_05": 0.25,
            "speedup_vs_full": 0.66,
        },
        "dense_to_shared_compact_to_xml_top5_keep06": {
            "wall_s": _wall_s(10.46),
            "vcmr_r1_05": 1.17,
            "speedup_vs_full": 0.70,
        },
    }
    return xml_runs, hybrid_runs


def main() -> None:
    xml_runs, hybrid_runs = _load_runs()
    num_queries = NUM_QUERIES
    full_wall_s = xml_runs["full"]["wall_s"]
    ours_rows = [
        ("Full XML", xml_runs["full"]["wall_s"] * 1000.0 / num_queries, xml_runs["full"]["vcmr_0.5_r1"], "1.00x"),
        ("Dense->XML top-1", xml_runs["top1"]["wall_s"] * 1000.0 / num_queries, xml_runs["top1"]["vcmr_0.5_r1"], f"{full_wall_s / xml_runs['top1']['wall_s']:.2f}x"),
        ("Dense->XML top-5", xml_runs["top5"]["wall_s"] * 1000.0 / num_queries, xml_runs["top5"]["vcmr_0.5_r1"], f"{full_wall_s / xml_runs['top5']['wall_s']:.2f}x"),
        ("Dense->XML top-10", xml_runs["top10"]["wall_s"] * 1000.0 / num_queries, xml_runs["top10"]["vcmr_0.5_r1"], f"{full_wall_s / xml_runs['top10']['wall_s']:.2f}x"),
    ]
    compaction_rows = [
        (
            "Shared compact XML",
            hybrid_runs["shared_compact_to_xml_keep06"]["wall_s"] * 1000.0 / num_queries,
            hybrid_runs["shared_compact_to_xml_keep06"]["vcmr_r1_05"],
            f"{hybrid_runs['shared_compact_to_xml_keep06']['speedup_vs_full']:.2f}x",
        ),
        (
            "Dense->compact->XML top-5",
            hybrid_runs["dense_to_shared_compact_to_xml_top5_keep06"]["wall_s"] * 1000.0 / num_queries,
            hybrid_runs["dense_to_shared_compact_to_xml_top5_keep06"]["vcmr_r1_05"],
            f"{hybrid_runs['dense_to_shared_compact_to_xml_top5_keep06']['speedup_vs_full']:.2f}x",
        ),
    ]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(6.6, 4.2))

    line_x = [row[1] for row in ours_rows]
    line_y = [row[2] for row in ours_rows]
    ax.plot(line_x, line_y, color="#0F766E", linewidth=1.5, alpha=0.75, zorder=1)

    point_specs = {
        "Full XML": {"color": "#0F172A", "marker": "o", "size": 72},
        "Dense->XML top-1": {"color": "#0F766E", "marker": "D", "size": 72},
        "Dense->XML top-5": {"color": "#14B8A6", "marker": "*", "size": 140},
        "Dense->XML top-10": {"color": "#2DD4BF", "marker": "^", "size": 90},
        "Shared compact XML": {"color": "#C2410C", "marker": "s", "size": 76},
        "Dense->compact->XML top-5": {"color": "#EA580C", "marker": "P", "size": 82},
    }

    for label, x, y, _speedup in ours_rows:
        spec = point_specs[label]
        ax.scatter(
            x,
            y,
            s=spec["size"],
            color=spec["color"],
            marker=spec["marker"],
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
        )

    for label, x, y, _speedup in compaction_rows:
        spec = point_specs[label]
        ax.scatter(
            x,
            y,
            s=spec["size"],
            color=spec["color"],
            marker=spec["marker"],
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
        )

    ax.annotate(
        "Selected: top-5\n1.48x faster",
        xy=(ours_rows[2][1], ours_rows[2][2]),
        xytext=(6.0, 1.34),
        textcoords="data",
        fontsize=8,
        color="#0F172A",
        weight="bold",
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "#0F172A"},
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#94A3B8", "alpha": 0.95},
        zorder=4,
    )

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=point_specs["Full XML"]["color"], markeredgecolor="black", markersize=6.5, label="Full XML"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=point_specs["Dense->XML top-1"]["color"], markeredgecolor="black", markersize=6.0, label="Dense->XML top-1"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor=point_specs["Dense->XML top-5"]["color"], markeredgecolor="black", markersize=9.0, label="Dense->XML top-5 (selected)"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=point_specs["Dense->XML top-10"]["color"], markeredgecolor="black", markersize=7.0, label="Dense->XML top-10"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=point_specs["Shared compact XML"]["color"], markeredgecolor="black", markersize=6.5, label="Shared compact XML"),
        Line2D([0], [0], marker="P", color="none", markerfacecolor=point_specs["Dense->compact->XML top-5"]["color"], markeredgecolor="black", markersize=7.0, label="Dense->compact->XML top-5"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=2,
        frameon=True,
        fontsize=7.2,
        columnspacing=1.1,
        handletextpad=0.4,
    )

    ax.set_title("Fixed-Feature Validation Trade-off for XML Variants", fontsize=10, weight="bold")
    ax.set_xlabel("Validation runtime (ms/query)")
    ax.set_ylabel(r"VCMR R@1, IoU=0.5 (\%)")
    ax.set_xlim(4.5, 11.7)
    ax.set_ylim(0.15, 1.75)

    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
