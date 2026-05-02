#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
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
    build_bm25_index,
    build_dense_index,
    fuse_ranks_rrf,
    fuse_scores_weighted,
    rerank_bm25_candidates_with_dense,
    score_windows_bm25,
    score_windows_dense,
)
from src.utils import Timer, ensure_dir, read_jsonl, write_json, write_jsonl


def _load_windows_by_video(windows_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
    windows_by_video: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(Path(windows_dir).glob("*.jsonl")):
        rows = read_jsonl(path)
        if rows:
            windows_by_video[path.stem] = rows
    return windows_by_video


def _parse_float_list(payload: str) -> list[float]:
    return [float(token.strip()) for token in payload.split(",") if token.strip()]


def _parse_int_list(payload: str) -> list[int]:
    return [int(token.strip()) for token in payload.split(",") if token.strip()]


def _make_prediction_row(
    *,
    query: dict[str, Any],
    retrieved_windows: list[dict[str, Any]],
    bm25_time_ms: float,
    dense_time_ms: float,
    fusion_time_ms: float,
) -> dict[str, Any]:
    total_retrieval_time_ms = bm25_time_ms + dense_time_ms + fusion_time_ms
    return {
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
        "bm25_time_ms": bm25_time_ms,
        "dense_time_ms": dense_time_ms,
        "fusion_time_ms": fusion_time_ms,
        "total_retrieval_time_ms": total_retrieval_time_ms,
        "retrieval_time_ms": total_retrieval_time_ms,
    }


def _write_run(
    *,
    output_dir: Path,
    run_name: str,
    predictions: list[dict[str, Any]],
    run_config: dict[str, Any],
) -> dict[str, Any]:
    predictions_path = output_dir / f"predictions_{run_name}.jsonl"
    summary_path = output_dir / f"summary_{run_name}.json"
    write_jsonl(predictions_path, predictions)
    summary = evaluate_predictions(predictions)
    summary.update(run_config)
    summary["predictions_path"] = str(predictions_path)
    write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BM25+dense fusion baselines on TVR.")
    parser.add_argument("--data-dir", default="data/tvr")
    parser.add_argument("--split", default="val")
    parser.add_argument("--query-path", default="")
    parser.add_argument("--windows-dir", default="data/tvr/processed/windows")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/tvr/fusion")
    parser.add_argument("--dense-model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--dense-batch-size", type=int, default=64)
    parser.add_argument("--normalization", choices=("minmax", "zscore"), default="minmax")
    parser.add_argument("--alphas", default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--rrf-ks", default="60")
    parser.add_argument("--bm25-topn-rerank", default="10,20,50")
    args = parser.parse_args()

    queries = load_tvr_queries(split=args.split, data_dir=args.data_dir, query_path=args.query_path)
    if args.max_examples > 0:
        queries = queries[: args.max_examples]
    windows_by_video = _load_windows_by_video(args.windows_dir)

    alphas = _parse_float_list(args.alphas)
    rrf_ks = _parse_int_list(args.rrf_ks)
    rerank_topns = _parse_int_list(args.bm25_topn_rerank)
    max_rerank_topn = max(rerank_topns) if rerank_topns else 0

    dense_retriever = DenseTextRetriever(
        model_name=args.dense_model_name,
        batch_size=args.dense_batch_size,
    )
    with Timer() as timer:
        query_embeddings = dense_retriever.encode_queries([str(query["query"]) for query in queries])
    avg_dense_query_ms = (timer.elapsed_s * 1000.0 / len(queries)) if queries else 0.0

    bm25_cache: dict[str, tuple[list[dict[str, Any]], Any]] = {}
    dense_cache: dict[str, tuple[list[dict[str, Any]], Any]] = {}

    weighted_predictions: dict[float, list[dict[str, Any]]] = {alpha: [] for alpha in alphas}
    rrf_predictions: dict[int, list[dict[str, Any]]] = {rrf_k: [] for rrf_k in rrf_ks}
    rerank_predictions: dict[int, list[dict[str, Any]]] = {top_n: [] for top_n in rerank_topns}

    for query, query_embedding in tqdm(
        zip(queries, query_embeddings, strict=True),
        total=len(queries),
        desc="TVR Fusion",
    ):
        vid_name = str(query["vid_name"])
        windows = windows_by_video.get(vid_name, [])
        if not windows:
            for alpha in alphas:
                weighted_predictions[alpha].append(
                    _make_prediction_row(
                        query=query,
                        retrieved_windows=[],
                        bm25_time_ms=0.0,
                        dense_time_ms=avg_dense_query_ms,
                        fusion_time_ms=0.0,
                    )
                )
            for rrf_k in rrf_ks:
                rrf_predictions[rrf_k].append(
                    _make_prediction_row(
                        query=query,
                        retrieved_windows=[],
                        bm25_time_ms=0.0,
                        dense_time_ms=avg_dense_query_ms,
                        fusion_time_ms=0.0,
                    )
                )
            for top_n in rerank_topns:
                rerank_predictions[top_n].append(
                    _make_prediction_row(
                        query=query,
                        retrieved_windows=[],
                        bm25_time_ms=0.0,
                        dense_time_ms=avg_dense_query_ms,
                        fusion_time_ms=0.0,
                    )
                )
            continue

        if vid_name not in bm25_cache:
            bm25_cache[vid_name] = (windows, build_bm25_index(windows)[0])
        if vid_name not in dense_cache:
            dense_cache[vid_name] = (windows, build_dense_index(windows, retriever=dense_retriever))

        _, bm25 = bm25_cache[vid_name]
        _, dense_index = dense_cache[vid_name]

        with Timer() as bm25_timer:
            bm25_scores = score_windows_bm25(query=str(query["query"]), bm25=bm25)
        bm25_time_ms = bm25_timer.elapsed_s * 1000.0

        with Timer() as dense_timer:
            dense_scores_full = score_windows_dense(
                document_embeddings=dense_index,
                query_embedding=query_embedding,
            )
        dense_time_full_ms = (dense_timer.elapsed_s * 1000.0) + avg_dense_query_ms
        dense_scores_list = dense_scores_full.tolist()

        for alpha in alphas:
            with Timer() as fusion_timer:
                retrieved_windows = fuse_scores_weighted(
                    windows=windows,
                    bm25_scores=bm25_scores,
                    dense_scores=dense_scores_list,
                    alpha=alpha,
                    normalization=args.normalization,
                    top_k=args.top_k,
                )
            weighted_predictions[alpha].append(
                _make_prediction_row(
                    query=query,
                    retrieved_windows=retrieved_windows,
                    bm25_time_ms=bm25_time_ms,
                    dense_time_ms=dense_time_full_ms,
                    fusion_time_ms=fusion_timer.elapsed_s * 1000.0,
                )
            )

        for rrf_k in rrf_ks:
            with Timer() as fusion_timer:
                retrieved_windows = fuse_ranks_rrf(
                    windows=windows,
                    bm25_scores=bm25_scores,
                    dense_scores=dense_scores_list,
                    rrf_k=rrf_k,
                    top_k=args.top_k,
                )
            rrf_predictions[rrf_k].append(
                _make_prediction_row(
                    query=query,
                    retrieved_windows=retrieved_windows,
                    bm25_time_ms=bm25_time_ms,
                    dense_time_ms=dense_time_full_ms,
                    fusion_time_ms=fusion_timer.elapsed_s * 1000.0,
                )
            )

        if rerank_topns:
            bm25_order = sorted(range(len(windows)), key=lambda idx: bm25_scores[idx], reverse=True)[:max_rerank_topn]
            topn_dense_scores: dict[int, float] = {}
            with Timer() as dense_subset_timer:
                if bm25_order:
                    candidate_embeddings = dense_index[bm25_order]
                    candidate_scores = score_windows_dense(
                        document_embeddings=candidate_embeddings,
                        query_embedding=query_embedding,
                    ).tolist()
                    topn_dense_scores = {
                        idx: float(score) for idx, score in zip(bm25_order, candidate_scores, strict=True)
                    }
            dense_subset_time_ms = (dense_subset_timer.elapsed_s * 1000.0) + avg_dense_query_ms

            for top_n in rerank_topns:
                bm25_scores_subset = bm25_scores
                dense_scores_subset = [
                    topn_dense_scores.get(idx, float("-inf")) for idx in range(len(windows))
                ]
                with Timer() as fusion_timer:
                    retrieved_windows = rerank_bm25_candidates_with_dense(
                        windows=windows,
                        bm25_scores=bm25_scores_subset,
                        dense_scores=dense_scores_subset,
                        top_n=top_n,
                        top_k=args.top_k,
                    )
                rerank_predictions[top_n].append(
                    _make_prediction_row(
                        query=query,
                        retrieved_windows=retrieved_windows,
                        bm25_time_ms=bm25_time_ms,
                        dense_time_ms=dense_subset_time_ms,
                        fusion_time_ms=fusion_timer.elapsed_s * 1000.0,
                    )
                )

    output_dir = ensure_dir(args.output_dir)
    comparison_rows: list[dict[str, Any]] = []

    for alpha in alphas:
        run_name = f"weighted_alpha_{str(alpha).replace('.', 'p')}"
        summary = _write_run(
            output_dir=output_dir,
            run_name=run_name,
            predictions=weighted_predictions[alpha],
            run_config={
                "split": args.split,
                "baseline": "bm25_dense_weighted_fusion",
                "fusion_mode": "weighted",
                "alpha": alpha,
                "normalization": args.normalization,
            },
        )
        comparison_rows.append(
            {
                "run_name": run_name,
                "baseline": "weighted",
                "alpha": alpha,
                "rrf_k": None,
                "top_n": None,
                "mean_iou": summary["mean_iou"],
                "recall_at_1_iou_0_3": summary["recall_at_1_iou_0_3"],
                "recall_at_1_iou_0_5": summary["recall_at_1_iou_0_5"],
                "recall_at_3_iou_0_3": summary["recall_at_3_iou_0_3"],
                "recall_at_3_iou_0_5": summary["recall_at_3_iou_0_5"],
                "avg_retrieval_latency_ms": summary["avg_retrieval_latency_ms"],
                "avg_bm25_time_ms": summary["avg_bm25_time_ms"],
                "avg_dense_time_ms": summary["avg_dense_time_ms"],
                "avg_fusion_time_ms": summary["avg_fusion_time_ms"],
            }
        )

    for rrf_k in rrf_ks:
        run_name = f"rrf_k_{rrf_k}"
        summary = _write_run(
            output_dir=output_dir,
            run_name=run_name,
            predictions=rrf_predictions[rrf_k],
            run_config={
                "split": args.split,
                "baseline": "bm25_dense_rank_fusion",
                "fusion_mode": "rrf",
                "rrf_k": rrf_k,
            },
        )
        comparison_rows.append(
            {
                "run_name": run_name,
                "baseline": "rrf",
                "alpha": None,
                "rrf_k": rrf_k,
                "top_n": None,
                "mean_iou": summary["mean_iou"],
                "recall_at_1_iou_0_3": summary["recall_at_1_iou_0_3"],
                "recall_at_1_iou_0_5": summary["recall_at_1_iou_0_5"],
                "recall_at_3_iou_0_3": summary["recall_at_3_iou_0_3"],
                "recall_at_3_iou_0_5": summary["recall_at_3_iou_0_5"],
                "avg_retrieval_latency_ms": summary["avg_retrieval_latency_ms"],
                "avg_bm25_time_ms": summary["avg_bm25_time_ms"],
                "avg_dense_time_ms": summary["avg_dense_time_ms"],
                "avg_fusion_time_ms": summary["avg_fusion_time_ms"],
            }
        )

    for top_n in rerank_topns:
        run_name = f"bm25_top{top_n}_dense_rerank"
        summary = _write_run(
            output_dir=output_dir,
            run_name=run_name,
            predictions=rerank_predictions[top_n],
            run_config={
                "split": args.split,
                "baseline": "bm25_topn_dense_rerank",
                "fusion_mode": "bm25_topn_dense_rerank",
                "top_n": top_n,
            },
        )
        comparison_rows.append(
            {
                "run_name": run_name,
                "baseline": "bm25_topn_dense_rerank",
                "alpha": None,
                "rrf_k": None,
                "top_n": top_n,
                "mean_iou": summary["mean_iou"],
                "recall_at_1_iou_0_3": summary["recall_at_1_iou_0_3"],
                "recall_at_1_iou_0_5": summary["recall_at_1_iou_0_5"],
                "recall_at_3_iou_0_3": summary["recall_at_3_iou_0_3"],
                "recall_at_3_iou_0_5": summary["recall_at_3_iou_0_5"],
                "avg_retrieval_latency_ms": summary["avg_retrieval_latency_ms"],
                "avg_bm25_time_ms": summary["avg_bm25_time_ms"],
                "avg_dense_time_ms": summary["avg_dense_time_ms"],
                "avg_fusion_time_ms": summary["avg_fusion_time_ms"],
            }
        )

    comparison_rows.sort(
        key=lambda row: (
            -float(row["mean_iou"] or 0.0),
            -float(row["recall_at_1_iou_0_3"] or 0.0),
        )
    )
    write_json(output_dir / "comparison.json", {"runs": comparison_rows})
    (output_dir / "comparison_table.md").write_text(
        "\n".join(
            [
                "| run_name | baseline | mean_iou | R@1@0.3 | R@1@0.5 | R@3@0.3 | R@3@0.5 | avg_latency_ms |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                *[
                    "| {run_name} | {baseline} | {mean_iou:.4f} | {recall_at_1_iou_0_3:.4f} | {recall_at_1_iou_0_5:.4f} | {recall_at_3_iou_0_3:.4f} | {recall_at_3_iou_0_5:.4f} | {avg_retrieval_latency_ms:.4f} |".format(
                        **row
                    )
                    for row in comparison_rows
                ],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(output_dir)


if __name__ == "__main__":
    main()
