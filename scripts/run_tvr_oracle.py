#!/usr/bin/env python3

from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys
from typing import Any

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import load_tvr_queries
from src.eval import compute_iou, evaluate_predictions
from src.utils import ensure_dir, read_jsonl, write_json, write_jsonl


def _load_windows_by_video(windows_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
    windows_by_video: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(Path(windows_dir).glob("*.jsonl")):
        rows = read_jsonl(path)
        if rows:
            windows_by_video[path.stem] = rows
    return windows_by_video


def _full_video_window(query: dict[str, Any]) -> dict[str, Any]:
    return {
        "vid_name": query["vid_name"],
        "window_id": f"{query['vid_name']}:full",
        "start_time": 0.0,
        "end_time": float(query.get("duration") or 0.0),
        "text": "",
        "score": 1.0,
        "rank": 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sanity/oracle TVR baselines.")
    parser.add_argument("--data-dir", default="data/tvr")
    parser.add_argument("--split", default="val")
    parser.add_argument("--query-path", default="")
    parser.add_argument("--windows-dir", default="data/tvr/processed/windows")
    parser.add_argument("--baseline", choices=["random_window", "full_video", "oracle_window"], default="oracle_window")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output-dir", default="outputs/tvr/oracle")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    queries = load_tvr_queries(split=args.split, data_dir=args.data_dir, query_path=args.query_path)
    windows_by_video = _load_windows_by_video(args.windows_dir)

    predictions: list[dict[str, Any]] = []
    for query in tqdm(queries, desc=f"TVR {args.baseline}"):
        windows = windows_by_video.get(str(query["vid_name"]), [])
        if args.baseline == "full_video" or not windows:
            retrieved_windows = [_full_video_window(query)]
        elif args.baseline == "random_window":
            sampled = windows[:]
            rng.shuffle(sampled)
            retrieved_windows = [{**window, "score": 0.0, "rank": idx + 1} for idx, window in enumerate(sampled[: args.top_k])]
        else:
            if query["gold_start_time"] is None or query["gold_end_time"] is None:
                retrieved_windows = [_full_video_window(query)]
                predictions.append(
                    {
                        "query_id": query["query_id"],
                        "desc_id": query["desc_id"],
                        "vid_name": query["vid_name"],
                        "query": query["query"],
                        "query_type": query["query_type"],
                        "gold_start_time": query["gold_start_time"],
                        "gold_end_time": query["gold_end_time"],
                        "predicted_start_time": retrieved_windows[0]["start_time"],
                        "predicted_end_time": retrieved_windows[0]["end_time"],
                        "retrieved_windows": retrieved_windows,
                        "retrieval_time_ms": 0.0,
                    }
                )
                continue
            scored = sorted(
                windows,
                key=lambda window: compute_iou(
                    float(window["start_time"]),
                    float(window["end_time"]),
                    float(query["gold_start_time"]),
                    float(query["gold_end_time"]),
                ),
                reverse=True,
            )
            retrieved_windows = [
                {**window, "score": 0.0, "rank": idx + 1}
                for idx, window in enumerate(scored[: args.top_k])
            ]

        predictions.append(
            {
                "query_id": query["query_id"],
                "desc_id": query["desc_id"],
                "vid_name": query["vid_name"],
                "query": query["query"],
                "query_type": query["query_type"],
                "gold_start_time": query["gold_start_time"],
                "gold_end_time": query["gold_end_time"],
                "predicted_start_time": retrieved_windows[0]["start_time"] if retrieved_windows else None,
                "predicted_end_time": retrieved_windows[0]["end_time"] if retrieved_windows else None,
                "retrieved_windows": retrieved_windows,
                "retrieval_time_ms": 0.0,
            }
        )

    output_dir = ensure_dir(args.output_dir)
    predictions_path = output_dir / f"{args.baseline}_{args.split}.jsonl"
    summary_path = output_dir / f"{args.baseline}_{args.split}.json"
    write_jsonl(predictions_path, predictions)
    summary = evaluate_predictions(predictions)
    summary.update(
        {
            "split": args.split,
            "baseline": args.baseline,
            "predictions_path": str(predictions_path),
        }
    )
    write_json(summary_path, summary)
    print(summary_path)


if __name__ == "__main__":
    main()
