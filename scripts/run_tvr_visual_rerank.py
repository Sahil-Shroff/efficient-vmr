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
from src.retrieval import (
    DenseTextRetriever,
    build_dense_index,
    rerank_windows_with_visual,
    retrieve_top_k_windows_from_embedding,
)
from src.utils import Timer, ensure_dir, read_jsonl, write_json, write_jsonl
from src.visual import ClipFeatureEncoder, load_visual_feature_file


def _load_windows_by_video(windows_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
    windows_by_video: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(Path(windows_dir).glob("*.jsonl")):
        rows = read_jsonl(path)
        if rows:
            windows_by_video[path.stem] = rows
    return windows_by_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerank dense subtitle windows with CLIP visual features on TVR.")
    parser.add_argument("--data-dir", default="data/tvr")
    parser.add_argument("--split", default="val")
    parser.add_argument("--query-path", default="")
    parser.add_argument("--windows-dir", default="data/tvr/processed/windows")
    parser.add_argument("--visual-features-dir", default="data/tvr/processed/visual_features")
    parser.add_argument("--candidate-top-k", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/tvr/visual_rerank")
    parser.add_argument("--text-model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--visual-model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default="")
    parser.add_argument("--text-batch-size", type=int, default=64)
    parser.add_argument("--visual-text-batch-size", type=int, default=32)
    parser.add_argument("--text-weight", type=float, default=0.6)
    parser.add_argument("--visual-weight", type=float, default=0.4)
    parser.add_argument("--visual-pool", choices=("max", "mean"), default="max")
    parser.add_argument("--apply-query-types", default="v")
    args = parser.parse_args()

    queries = load_tvr_queries(split=args.split, data_dir=args.data_dir, query_path=args.query_path)
    if args.max_examples > 0:
        queries = queries[: args.max_examples]
    windows_by_video = _load_windows_by_video(args.windows_dir)
    apply_query_types = {token.strip() for token in args.apply_query_types.split(",") if token.strip()}

    text_retriever = DenseTextRetriever(
        model_name=args.text_model_name,
        device=args.device or None,
        batch_size=args.text_batch_size,
    )
    visual_encoder = ClipFeatureEncoder(
        model_name=args.visual_model_name,
        device=args.device or None,
        batch_size=args.visual_text_batch_size,
    )

    with Timer() as timer:
        text_query_embeddings = text_retriever.encode_queries([str(query["query"]) for query in queries])
    avg_text_query_ms = (timer.elapsed_s * 1000.0 / len(queries)) if queries else 0.0

    with Timer() as timer:
        visual_query_embeddings = visual_encoder.encode_texts([str(query["query"]) for query in queries])
    avg_visual_query_ms = (timer.elapsed_s * 1000.0 / len(queries)) if queries else 0.0

    dense_cache: dict[str, tuple[list[dict[str, Any]], Any]] = {}
    visual_cache: dict[str, tuple[Any, Any] | None] = {}
    predictions: list[dict[str, Any]] = []
    candidate_top_k = max(args.candidate_top_k, args.top_k)

    for query, text_query_embedding, visual_query_embedding in tqdm(
        zip(queries, text_query_embeddings, visual_query_embeddings, strict=True),
        total=len(queries),
        desc="TVR Visual Rerank",
    ):
        vid_name = str(query["vid_name"])
        windows = windows_by_video.get(vid_name, [])
        if vid_name not in dense_cache and windows:
            dense_cache[vid_name] = (windows, build_dense_index(windows, retriever=text_retriever))

        with Timer() as timer:
            if windows:
                cached_windows, dense_index = dense_cache[vid_name]
                candidate_windows = retrieve_top_k_windows_from_embedding(
                    windows=cached_windows,
                    document_embeddings=dense_index,
                    query_embedding=text_query_embedding,
                    top_k=candidate_top_k,
                )
            else:
                candidate_windows = []

            used_visual = False
            retrieved_windows = candidate_windows[: args.top_k]
            if candidate_windows and str(query["query_type"]) in apply_query_types:
                if vid_name not in visual_cache:
                    feature_path = Path(args.visual_features_dir) / f"{vid_name}.npz"
                    visual_cache[vid_name] = load_visual_feature_file(feature_path) if feature_path.exists() else None
                visual_features = visual_cache.get(vid_name)
                if visual_features is not None:
                    frame_timestamps, frame_embeddings = visual_features
                    retrieved_windows = rerank_windows_with_visual(
                        windows=candidate_windows,
                        query_embedding=visual_query_embedding,
                        frame_timestamps=frame_timestamps,
                        frame_embeddings=frame_embeddings,
                        text_weight=args.text_weight,
                        visual_weight=args.visual_weight,
                        pool=args.visual_pool,
                    )[: args.top_k]
                    used_visual = True
            else:
                used_visual = False

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
                "used_visual_rerank": used_visual,
                "retrieval_time_ms": (timer.elapsed_s * 1000.0) + avg_text_query_ms + avg_visual_query_ms,
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
            "baseline": "dense_visual_rerank",
            "text_model_name": args.text_model_name,
            "visual_model_name": args.visual_model_name,
            "visual_features_dir": args.visual_features_dir,
            "text_weight": args.text_weight,
            "visual_weight": args.visual_weight,
            "apply_query_types": sorted(apply_query_types),
            "predictions_path": str(predictions_path),
        }
    )
    write_json(summary_path, summary)
    print(summary_path)


if __name__ == "__main__":
    main()
