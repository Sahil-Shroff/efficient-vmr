from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ..data import load_flattened_examples
from ..retrieval import build_bm25_index, retrieve_top_k_windows
from ..utils import Timer, ensure_dir, read_jsonl, write_json


def compute_iou(pred_start: float, pred_end: float, gold_start: float, gold_end: float) -> float:
    intersection = max(0.0, min(pred_end, gold_end) - max(pred_start, gold_start))
    union = max(pred_end, gold_end) - min(pred_start, gold_start)
    if union <= 0:
        return 0.0
    return intersection / union


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _latency_percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, math.floor(percentile * len(sorted_values)))
    return sorted_values[index]


def _load_windows_by_video(windows_dir: Path) -> dict[str, list[dict[str, Any]]]:
    windows_by_video: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(windows_dir.glob("*.jsonl")):
        windows = read_jsonl(path)
        if windows:
            windows_by_video[path.stem] = windows
    return windows_by_video


def _filter_examples_by_video_ids(
    examples: list[dict[str, Any]],
    *,
    video_ids: set[str] | None = None,
    max_videos: int = 0,
) -> list[dict[str, Any]]:
    if video_ids is not None:
        examples = [example for example in examples if str(example.get("video_id", "")) in video_ids]

    if max_videos <= 0:
        return examples

    selected_video_ids: set[str] = set()
    filtered_examples: list[dict[str, Any]] = []
    for example in examples:
        video_id = str(example.get("video_id", ""))
        if not video_id:
            continue
        if video_id in selected_video_ids or len(selected_video_ids) < max_videos:
            selected_video_ids.add(video_id)
            filtered_examples.append(example)
    return filtered_examples


def _load_requested_video_ids(path: str | None) -> set[str] | None:
    if not path:
        return None

    requested_video_ids: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            video_id = line.strip()
            if video_id:
                requested_video_ids.add(video_id)
    return requested_video_ids


def _load_gold_spans(path: str | None) -> dict[str, dict[str, float]]:
    if not path:
        return {}

    gold_spans: dict[str, dict[str, float]] = {}
    for row in read_jsonl(path):
        question_id = str(row.get("question_id", "")).strip()
        if not question_id:
            continue

        start_time = row.get("start_time", row.get("gold_start_time"))
        end_time = row.get("end_time", row.get("gold_end_time"))
        if start_time is None or end_time is None:
            continue

        gold_spans[question_id] = {
            "start_time": float(start_time),
            "end_time": float(end_time),
        }
    return gold_spans


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a subtitle-based temporal retrieval baseline for Video Moment Retrieval.")
    parser.add_argument("--split", default="Video_MMLU")
    parser.add_argument("--windows-dir", default="data/subtitles/windows")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output-dir", default="outputs/vmr")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--video-ids-path", default="")
    parser.add_argument("--gold-spans-path", default="")
    args = parser.parse_args()

    examples = load_flattened_examples(split=args.split)
    requested_video_ids = _load_requested_video_ids(args.video_ids_path or None)
    examples = _filter_examples_by_video_ids(
        examples,
        video_ids=requested_video_ids,
        max_videos=args.max_videos,
    )
    if args.max_examples > 0:
        examples = examples[: args.max_examples]
    if not examples:
        raise ValueError("No flattened examples available for evaluation.")

    windows_by_video = _load_windows_by_video(Path(args.windows_dir))
    gold_spans_by_question_id = _load_gold_spans(args.gold_spans_path or None)
    index_cache: dict[str, tuple[list[dict[str, Any]], Any]] = {}
    out_dir = ensure_dir(args.output_dir)
    predictions_path = out_dir / f"predictions_{args.split}.jsonl"
    summary_path = out_dir / f"summary_{args.split}.json"

    retrieval_latencies: list[float] = []
    ious_at_1: list[float] = []
    start_errors: list[float] = []
    end_errors: list[float] = []
    recall_hits: dict[tuple[int, float], int] = {(1, 0.3): 0, (1, 0.5): 0, (3, 0.3): 0, (3, 0.5): 0}

    num_examples_with_windows = 0
    num_examples_with_gold_spans = 0
    predictions: list[dict[str, Any]] = []
    evaluated_video_ids = {str(example.get("video_id", "")) for example in examples if example.get("video_id")}

    for example in tqdm(examples, desc="VMR baseline"):
        video_id = str(example.get("video_id", ""))
        question = str(example.get("question", ""))
        question_id = str(example.get("question_id", ""))
        gold_span = gold_spans_by_question_id.get(question_id)
        gold_start = gold_span["start_time"] if gold_span else example.get("start_time")
        gold_end = gold_span["end_time"] if gold_span else example.get("end_time")
        windows = windows_by_video.get(video_id, [])

        if video_id not in index_cache and windows:
            index_cache[video_id] = (windows, build_bm25_index(windows)[0])

        with Timer() as retrieval_timer:
            if windows:
                cached_windows, bm25 = index_cache[video_id]
                retrieved = retrieve_top_k_windows(
                    query=question,
                    windows=cached_windows,
                    bm25=bm25,
                    top_k=args.top_k,
                )
            else:
                retrieved = []
        retrieval_time_ms = retrieval_timer.elapsed_s * 1000.0
        retrieval_latencies.append(retrieval_time_ms)

        if retrieved:
            num_examples_with_windows += 1
            top_prediction = retrieved[0]
            predicted_start = top_prediction["start_time"]
            predicted_end = top_prediction["end_time"]
        else:
            predicted_start = None
            predicted_end = None

        row: dict[str, Any] = {
            "video_id": video_id,
            "question_id": question_id,
            "qa_type": example.get("qa_type"),
            "question": question,
            "predicted_start_time": predicted_start,
            "predicted_end_time": predicted_end,
            "gold_start_time": gold_start,
            "gold_end_time": gold_end,
            "retrieval_time_ms": retrieval_time_ms,
            "retrieved_windows": retrieved,
        }

        if gold_start is not None and gold_end is not None and retrieved:
            num_examples_with_gold_spans += 1
            top1_iou = compute_iou(
                float(predicted_start),
                float(predicted_end),
                float(gold_start),
                float(gold_end),
            )
            ious_at_1.append(top1_iou)
            start_errors.append(abs(float(predicted_start) - float(gold_start)))
            end_errors.append(abs(float(predicted_end) - float(gold_end)))

            row["top1_iou"] = top1_iou
            row["start_error"] = start_errors[-1]
            row["end_error"] = end_errors[-1]

            for k in (1, 3):
                topk = retrieved[: min(k, len(retrieved))]
                best_iou = max(
                    compute_iou(
                        float(window["start_time"]),
                        float(window["end_time"]),
                        float(gold_start),
                        float(gold_end),
                    )
                    for window in topk
                )
                row[f"best_iou_at_{k}"] = best_iou
                for threshold in (0.3, 0.5):
                    if best_iou >= threshold:
                        recall_hits[(k, threshold)] += 1

        predictions.append(row)

    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "split": args.split,
        "num_examples": len(examples),
        "num_videos": len(evaluated_video_ids),
        "num_examples_with_windows": num_examples_with_windows,
        "num_examples_with_gold_spans": num_examples_with_gold_spans,
        "avg_retrieval_time_ms": _mean(retrieval_latencies),
        "p50_retrieval_time_ms": _latency_percentile(retrieval_latencies, 0.50),
        "p95_retrieval_time_ms": _latency_percentile(retrieval_latencies, 0.95),
        "mean_top1_iou": _mean(ious_at_1),
        "mean_start_error": _mean(start_errors),
        "mean_end_error": _mean(end_errors),
        "recall_at_1_iou_0_3": None,
        "recall_at_1_iou_0_5": None,
        "recall_at_3_iou_0_3": None,
        "recall_at_3_iou_0_5": None,
        "gold_spans_path": args.gold_spans_path or None,
        "predictions_file": str(predictions_path),
    }
    if num_examples_with_gold_spans > 0:
        summary["recall_at_1_iou_0_3"] = recall_hits[(1, 0.3)] / num_examples_with_gold_spans
        summary["recall_at_1_iou_0_5"] = recall_hits[(1, 0.5)] / num_examples_with_gold_spans
        summary["recall_at_3_iou_0_3"] = recall_hits[(3, 0.3)] / num_examples_with_gold_spans
        summary["recall_at_3_iou_0_5"] = recall_hits[(3, 0.5)] / num_examples_with_gold_spans

    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
