#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "report" / "figures"
PDF_PATH = FIG_DIR / "pipeline_overview.pdf"
PNG_PATH = FIG_DIR / "pipeline_overview.png"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rounded_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
    face: str,
    edge: str,
    *,
    title_size: float = 10.5,
    body_size: float = 8.0,
) -> None:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.55,rounding_size=3.0",
        linewidth=1.6,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(box)
    ax.text(
        x + 1.2,
        y + h - 2.3,
        title,
        ha="left",
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color="#0F172A",
    )
    ax.text(
        x + w / 2.0,
        y + h / 2.0 - 0.5,
        body,
        ha="center",
        va="center",
        fontsize=body_size,
        color="#475569",
        linespacing=1.25,
    )


def _section_tag(ax, x: float, y: float, label: str, face: str, edge: str, text_color: str) -> None:
    tag = FancyBboxPatch(
        (x, y),
        16.0,
        4.2,
        boxstyle="round,pad=0.3,rounding_size=2.0",
        linewidth=1.2,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(tag)
    ax.text(x + 8.0, y + 2.1, label, ha="center", va="center", fontsize=8.7, fontweight="bold", color=text_color)


def _arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    lw: float = 1.9,
    style: str = "solid",
) -> None:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=lw,
        linestyle=style,
        color=color,
        shrinkA=2,
        shrinkB=2,
        connectionstyle="arc3,rad=0",
    )
    ax.add_patch(patch)


def _metric_card(ax, x: float, y: float, w: float, h: float, headline: str, sub: str, face: str, edge: str) -> None:
    card = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.55,rounding_size=3.0",
        linewidth=1.5,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(card)
    ax.text(
        x + w / 2.0,
        y + h * 0.62,
        headline,
        ha="center",
        va="center",
        fontsize=14.0,
        fontweight="bold",
        color="#0F172A",
    )
    ax.text(
        x + w / 2.0,
        y + h * 0.27,
        sub,
        ha="center",
        va="center",
        fontsize=7.7,
        color="#475569",
        linespacing=1.2,
    )


def main() -> None:
    xml_summary = _load_json(REPO_ROOT / "outputs" / "tvr" / "xml_hierarchical" / "dense_shortlist_xml_eval_summary.json")[
        "runs"
    ]
    hybrid_summary = {
        row["method"]: row
        for row in _load_json(REPO_ROOT / "outputs" / "tvr" / "xml_hybrid" / "hybrid_eval_summary.json")["runs"]
    }

    num_queries = 10895
    full_ms = xml_summary["full"]["wall_s"] * 1000.0 / num_queries
    ours_ms = xml_summary["top5"]["wall_s"] * 1000.0 / num_queries
    ours_speedup = full_ms / ours_ms
    full_vcmr = xml_summary["full"]["vcmr_0.5_r1"]
    ours_vcmr = xml_summary["top5"]["vcmr_0.5_r1"]
    full_svmr = xml_summary["full"]["svmr_0.5_r1"]
    ours_svmr = xml_summary["top5"]["svmr_0.5_r1"]
    full_candidates = int(hybrid_summary["full_xml"]["avg_candidates_per_query"])
    ours_candidates = int(hybrid_summary["dense_to_xml_top5"]["avg_candidates_per_query"])

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(14.0, 5.8))
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 100)
    ax.axis("off")

    offline_panel = FancyBboxPatch(
        (2.5, 64.0),
        115.0,
        26.0,
        boxstyle="round,pad=0.8,rounding_size=4.2",
        linewidth=1.2,
        edgecolor="#CBD5E1",
        facecolor="#F8FAFC",
    )
    online_panel = FancyBboxPatch(
        (2.5, 31.5),
        115.0,
        26.5,
        boxstyle="round,pad=0.8,rounding_size=4.2",
        linewidth=1.2,
        edgecolor="#C7D2FE",
        facecolor="#F8FBFF",
    )
    ax.add_patch(offline_panel)
    ax.add_patch(online_panel)

    ax.text(
        60.0,
        96.0,
        "Dense-Shortlist to XML Architecture",
        ha="center",
        va="center",
        fontsize=18.0,
        fontweight="bold",
        color="#0F172A",
    )
    ax.text(
        60.0,
        92.1,
        "Inspired by the late-fusion XML view in the TVR paper, but redrawn for the measured repo pipeline",
        ha="center",
        va="center",
        fontsize=9.0,
        color="#64748B",
    )

    _section_tag(ax, 5.5, 85.8, "Offline Assets", "#E2E8F0", "#CBD5E1", "#0F172A")
    _section_tag(ax, 5.5, 54.5, "Online Path", "#DBEAFE", "#BFDBFE", "#1D4ED8")

    _rounded_box(ax, 6.0, 70.0, 15.0, 13.0, "TVR subtitles", "jsonl dialogue\nsegments", "#FFF7ED", "#FDBA74")
    _rounded_box(ax, 24.0, 70.0, 15.0, 13.0, "Normalize", "trim whitespace\nmerge duplicates", "#FFF7ED", "#FDBA74")
    _rounded_box(ax, 42.0, 70.0, 16.0, 13.0, "Window builder", "12s windows\n6s stride", "#FEF3C7", "#FBBF24")
    _rounded_box(ax, 61.0, 70.0, 16.0, 13.0, "Per-video cache", "subtitle windows\nlazy index build", "#ECFCCB", "#84CC16")
    _rounded_box(ax, 82.0, 70.0, 23.0, 13.0, "Official XML features", "video / subtitle / query\nprecomputed HDF5 tensors", "#E0F2FE", "#38BDF8")

    _arrow(ax, (21.0, 76.5), (24.0, 76.5), color="#B45309")
    _arrow(ax, (39.0, 76.5), (42.0, 76.5), color="#B45309")
    _arrow(ax, (58.0, 76.5), (61.0, 76.5), color="#A16207")

    _rounded_box(ax, 6.0, 38.0, 15.0, 16.0, "User query", "natural-language\nmoment request", "#E2E8F0", "#94A3B8")
    _rounded_box(ax, 24.0, 38.0, 15.0, 16.0, "MiniLM encoder", "batch query\nembedding", "#DBEAFE", "#60A5FA")
    _rounded_box(ax, 43.0, 38.0, 18.0, 16.0, "Dense subtitle retrieval", "cosine similarity over\ncached windows\n0.39 ms front-end", "#CCFBF1", "#14B8A6")
    _rounded_box(ax, 64.0, 38.0, 16.0, 16.0, "Top-5 shortlist", "prune corpus to\nretained videos", "#CFFAFE", "#06B6D4")
    _rounded_box(ax, 83.5, 35.2, 24.0, 21.0, "", "", "#FEF3C7", "#F59E0B")
    _rounded_box(ax, 110.5, 40.2, 6.2, 11.0, "Output", "moment\n+ scores", "#DCFCE7", "#22C55E", title_size=8.4, body_size=7.0)

    _arrow(ax, (21.0, 46.0), (24.0, 46.0), color="#475569")
    _arrow(ax, (39.0, 46.0), (43.0, 46.0), color="#2563EB")
    _arrow(ax, (61.0, 46.0), (64.0, 46.0), color="#0F766E")
    _arrow(ax, (80.0, 46.0), (83.5, 46.0), color="#0891B2")
    _arrow(ax, (107.5, 46.0), (110.5, 46.0), color="#B45309")

    _arrow(ax, (69.5, 70.0), (52.5, 54.2), color="#65A30D", lw=1.7, style="dashed")
    _arrow(ax, (95.0, 70.0), (96.2, 56.2), color="#0284C7", lw=1.7, style="dashed")

    ax.text(95.5, 54.5, "XML localizer", ha="center", va="center", fontsize=10.4, fontweight="bold", color="#0F172A")

    pill_y = 47.8
    pills = [
        (86.0, 5.6, "Query", "#E2E8F0"),
        (92.2, 5.8, "Video", "#DBEAFE"),
        (98.8, 6.2, "Subtitle", "#FCE7F3"),
    ]
    for x, w, text, face in pills:
        pill = FancyBboxPatch(
            (x, pill_y),
            w,
            3.8,
            boxstyle="round,pad=0.25,rounding_size=1.5",
            linewidth=1.0,
            edgecolor="#CBD5E1",
            facecolor=face,
        )
        ax.add_patch(pill)
        ax.text(x + w / 2.0, pill_y + 1.9, text, ha="center", va="center", fontsize=7.3, color="#0F172A")

    xml_backbone = FancyBboxPatch(
        (86.6, 42.8),
        18.2,
        4.7,
        boxstyle="round,pad=0.25,rounding_size=1.8",
        linewidth=1.1,
        edgecolor="#F59E0B",
        facecolor="#FFFBEB",
    )
    convse = FancyBboxPatch(
        (86.6, 36.6),
        18.2,
        5.3,
        boxstyle="round,pad=0.25,rounding_size=1.8",
        linewidth=1.1,
        edgecolor="#F59E0B",
        facecolor="#FFFBEB",
    )
    ax.add_patch(xml_backbone)
    ax.add_patch(convse)
    ax.text(95.7, 45.15, "late-fusion XML backbone", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#92400E")
    ax.text(
        95.7,
        39.25,
        "ConvSE detector\n+ dynamic programming",
        ha="center",
        va="center",
        fontsize=7.5,
        color="#B45309",
        linespacing=1.15,
    )
    ax.text(95.5, 32.8, "shared video-subtitle timeline", ha="center", va="center", fontsize=7.7, color="#B45309")

    timeline = Rectangle((112.0, 42.4), 3.5, 1.1, linewidth=0.8, edgecolor="#64748B", facecolor="#E2E8F0")
    highlight = Rectangle((113.0, 42.5), 1.5, 0.9, linewidth=0.0, facecolor="#22C55E")
    ax.add_patch(timeline)
    ax.add_patch(highlight)

    ax.text(
        60.0,
        26.5,
        "Measured validation operating point for dense -> XML top-5",
        ha="center",
        va="center",
        fontsize=8.4,
        color="#64748B",
    )

    _metric_card(
        ax,
        6.0,
        7.0,
        24.0,
        14.0,
        f"{full_candidates} -> {ours_candidates}",
        "candidate videos per query\nbefore XML scoring",
        "#F8FAFC",
        "#CBD5E1",
    )
    _metric_card(
        ax,
        34.0,
        7.0,
        24.0,
        14.0,
        f"{ours_speedup:.2f}x",
        f"speedup vs full XML\n{full_ms:.2f} -> {ours_ms:.2f} ms/query",
        "#ECFDF5",
        "#86EFAC",
    )
    _metric_card(
        ax,
        62.0,
        7.0,
        24.0,
        14.0,
        f"{full_vcmr:.2f} -> {ours_vcmr:.2f}",
        "VCMR R@1@0.5\nvalidation improvement",
        "#EFF6FF",
        "#93C5FD",
    )
    _metric_card(
        ax,
        90.0,
        7.0,
        24.0,
        14.0,
        f"{full_svmr:.2f} = {ours_svmr:.2f}",
        "SVMR R@1@0.5\nlocalization preserved",
        "#FFF7ED",
        "#FDBA74",
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.5)
    fig.savefig(PDF_PATH, bbox_inches="tight")
    fig.savefig(PNG_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(PDF_PATH)


if __name__ == "__main__":
    main()
