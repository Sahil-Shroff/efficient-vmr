from __future__ import annotations

import math
from typing import Any


def compute_iou(pred_start: float, pred_end: float, gold_start: float, gold_end: float) -> float:
    intersection = max(0.0, min(pred_end, gold_end) - max(pred_start, gold_start))
    union = max(pred_end, gold_end) - min(pred_start, gold_start)
    if union <= 0.0:
        return 0.0
    return intersection / union


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, math.floor(percentile * len(sorted_values)))
    return sorted_values[index]


def _window_iou(window: dict[str, Any], row: dict[str, Any]) -> float:
    gold_start = row.get("gold_start_time")
    gold_end = row.get("gold_end_time")
    if gold_start is None or gold_end is None:
        return 0.0
    if str(window.get("vid_name", "")) != str(row.get("vid_name", "")):
        return 0.0
    return compute_iou(
        float(window["start_time"]),
        float(window["end_time"]),
        float(gold_start),
        float(gold_end),
    )


def _evaluate_subset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    retrieval_latencies: list[float] = []
    bm25_latencies: list[float] = []
    dense_latencies: list[float] = []
    fusion_latencies: list[float] = []
    total_retrieval_latencies: list[float] = []
    top1_ious: list[float] = []
    start_errors: list[float] = []
    end_errors: list[float] = []
    recall_hits: dict[tuple[int, float], int] = {(1, 0.3): 0, (1, 0.5): 0, (3, 0.3): 0, (3, 0.5): 0}

    num_examples_with_gold = 0
    num_examples_with_predictions = 0
    for row in rows:
        retrieval_time_ms = row.get("retrieval_time_ms", row.get("total_retrieval_time_ms"))
        if retrieval_time_ms is not None:
            retrieval_latencies.append(float(retrieval_time_ms))
        if row.get("bm25_time_ms") is not None:
            bm25_latencies.append(float(row["bm25_time_ms"]))
        if row.get("dense_time_ms") is not None:
            dense_latencies.append(float(row["dense_time_ms"]))
        if row.get("fusion_time_ms") is not None:
            fusion_latencies.append(float(row["fusion_time_ms"]))
        if row.get("total_retrieval_time_ms") is not None:
            total_retrieval_latencies.append(float(row["total_retrieval_time_ms"]))

        gold_start = row.get("gold_start_time")
        gold_end = row.get("gold_end_time")
        if gold_start is None or gold_end is None:
            continue
        num_examples_with_gold += 1

        retrieved_windows = row.get("retrieved_windows", []) or []
        if retrieved_windows:
            num_examples_with_predictions += 1
            top1_iou = _window_iou(retrieved_windows[0], row)
            top1_ious.append(top1_iou)
            start_errors.append(abs(float(retrieved_windows[0]["start_time"]) - float(gold_start)))
            end_errors.append(abs(float(retrieved_windows[0]["end_time"]) - float(gold_end)))

            for k in (1, 3):
                top_k_windows = retrieved_windows[: min(k, len(retrieved_windows))]
                best_iou = max(_window_iou(window, row) for window in top_k_windows)
                for threshold in (0.3, 0.5):
                    if best_iou >= threshold:
                        recall_hits[(k, threshold)] += 1

    summary: dict[str, Any] = {
        "num_examples": len(rows),
        "num_examples_with_gold": num_examples_with_gold,
        "num_examples_with_predictions": num_examples_with_predictions,
        "mean_iou": _mean(top1_ious),
        "mean_absolute_start_error": _mean(start_errors),
        "mean_absolute_end_error": _mean(end_errors),
        "recall_at_1_iou_0_3": None,
        "recall_at_1_iou_0_5": None,
        "recall_at_3_iou_0_3": None,
        "recall_at_3_iou_0_5": None,
        "avg_retrieval_latency_ms": _mean(retrieval_latencies),
        "p50_retrieval_latency_ms": _percentile(retrieval_latencies, 0.50),
        "p95_retrieval_latency_ms": _percentile(retrieval_latencies, 0.95),
        "avg_bm25_time_ms": _mean(bm25_latencies),
        "avg_dense_time_ms": _mean(dense_latencies),
        "avg_fusion_time_ms": _mean(fusion_latencies),
        "avg_total_retrieval_time_ms": _mean(total_retrieval_latencies),
        "p50_total_retrieval_time_ms": _percentile(total_retrieval_latencies, 0.50),
        "p95_total_retrieval_time_ms": _percentile(total_retrieval_latencies, 0.95),
    }
    if num_examples_with_gold > 0:
        summary["recall_at_1_iou_0_3"] = recall_hits[(1, 0.3)] / num_examples_with_gold
        summary["recall_at_1_iou_0_5"] = recall_hits[(1, 0.5)] / num_examples_with_gold
        summary["recall_at_3_iou_0_3"] = recall_hits[(3, 0.3)] / num_examples_with_gold
        summary["recall_at_3_iou_0_5"] = recall_hits[(3, 0.5)] / num_examples_with_gold
    return summary


def evaluate_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = _evaluate_subset(rows)
    summary["by_query_type"] = {}
    for query_type in ("v", "t", "vt"):
        type_rows = [row for row in rows if str(row.get("query_type", "")) == query_type]
        summary["by_query_type"][query_type] = _evaluate_subset(type_rows)
    return summary
