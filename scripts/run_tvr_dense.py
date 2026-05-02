#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import load_tvr_queries
from src.eval import evaluate_predictions
from src.retrieval import DenseTextRetriever, build_dense_index, retrieve_top_k_windows_from_embedding
from src.utils import Timer, ensure_dir, read_jsonl, write_json, write_jsonl


def _load_windows_by_video(windows_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
    windows_by_video: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(Path(windows_dir).glob("*.jsonl")):
        rows = read_jsonl(path)
        if rows:
            windows_by_video[path.stem] = rows
    return windows_by_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a dense subtitle-window retriever on TVR.")
    parser.add_argument("--data-dir", default="data/tvr")
    parser.add_argument("--split", default="val")
    parser.add_argument("--query-path", default="")
    parser.add_argument("--windows-dir", default="data/tvr/processed/windows")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/tvr/dense")
    parser.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", default="")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cache-folder", default="")
    args = parser.parse_args()

    queries = load_tvr_queries(split=args.split, data_dir=args.data_dir, query_path=args.query_path)
    if args.max_examples > 0:
        queries = queries[: args.max_examples]
    windows_by_video = _load_windows_by_video(args.windows_dir)
    retriever = DenseTextRetriever(
        model_name=args.model_name,
        device=args.device or None,
        batch_size=args.batch_size,
        cache_folder=args.cache_folder or None,
    )
    with Timer() as timer:
        query_embeddings = retriever.encode_queries([str(query["query"]) for query in queries])
    avg_query_encoding_ms = (timer.elapsed_s * 1000.0 / len(queries)) if queries else 0.0

    dense_cache: dict[str, tuple[list[dict[str, Any]], Any]] = {}
    predictions: list[dict[str, Any]] = []
    for query, query_embedding in tqdm(
        zip(queries, query_embeddings, strict=True),
        total=len(queries),
        desc="TVR Dense",
    ):
        vid_name = str(query["vid_name"])
        windows = windows_by_video.get(vid_name, [])
        if vid_name not in dense_cache and windows:
            dense_cache[vid_name] = (windows, build_dense_index(windows, retriever=retriever))

        with Timer() as timer:
            if windows:
                cached_windows, dense_index = dense_cache[vid_name]
                retrieved_windows = retrieve_top_k_windows_from_embedding(
                    windows=cached_windows,
                    document_embeddings=dense_index,
                    query_embedding=query_embedding,
                    top_k=args.top_k,
                )
            else:
                retrieved_windows = []

        predictions.append(
            {
                "query_id": query["query_id"],
                "desc_id": query["desc_id"],
                "vid_name": vid_name,
                "query": query["query"],
                "query_type": query["query_type"],
                "gold_start_time": query["gold_start_time"],
                "gold_end_time": query["gold_end_time"],
                "predicted_start_time": retrieved_windows[0]["start_time"] if retrieved_windows else None,
                "predicted_end_time": retrieved_windows[0]["end_time"] if retrieved_windows else None,
                "retrieved_windows": retrieved_windows,
                "retrieval_time_ms": (timer.elapsed_s * 1000.0) + avg_query_encoding_ms,
            }
        )

    output_dir = ensure_dir(args.output_dir)
    predictions_path = output_dir / f"predictions_{args.split}.jsonl"
    summary_path = output_dir / f"summary_{args.split}.json"
    write_jsonl(predictions_path, predictions)
    summary = evaluate_predictions(predictions)
    summary.update(
        {
            "split": args.split,
            "baseline": "dense_subtitle_windows",
            "model_name": args.model_name,
            "predictions_path": str(predictions_path),
        }
    )
    write_json(summary_path, summary)
    print(summary_path)


if __name__ == "__main__":
    main()
