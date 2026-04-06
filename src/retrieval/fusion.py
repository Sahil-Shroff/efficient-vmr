from __future__ import annotations

from typing import Any

import numpy as np


def _minmax_normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if max_score - min_score <= 1e-8:
        return [1.0 for _ in scores]
    return [(score - min_score) / (max_score - min_score) for score in scores]


def _zscore_normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    mean_score = sum(scores) / len(scores)
    variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
    std_score = variance ** 0.5
    if std_score <= 1e-8:
        return [0.0 for _ in scores]
    return [(score - mean_score) / std_score for score in scores]


def normalize_scores(scores: list[float], *, method: str = "minmax") -> list[float]:
    if method == "zscore":
        return _zscore_normalize(scores)
    if method == "minmax":
        return _minmax_normalize(scores)
    raise ValueError(f"Unsupported normalization method: {method}")


def rank_windows_by_scores(
    *,
    windows: list[dict[str, Any]],
    scores: list[float],
    top_k: int,
    score_key: str = "score",
    extra_fields: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if not windows:
        return []

    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]
    rows: list[dict[str, Any]] = []
    for rank, (idx, score) in enumerate(ranked, start=1):
        payload = {
            **windows[idx],
            score_key: float(score),
            "score": float(score),
            "rank": rank,
        }
        if extra_fields is not None:
            payload.update(extra_fields[idx])
        rows.append(payload)
    return rows


def fuse_scores_weighted(
    *,
    windows: list[dict[str, Any]],
    bm25_scores: list[float],
    dense_scores: list[float],
    alpha: float,
    normalization: str = "minmax",
    top_k: int,
) -> list[dict[str, Any]]:
    bm25_scores_norm = normalize_scores(bm25_scores, method=normalization)
    dense_scores_norm = normalize_scores(dense_scores, method=normalization)

    fused_scores = [
        (alpha * bm25_score_norm) + ((1.0 - alpha) * dense_score_norm)
        for bm25_score_norm, dense_score_norm in zip(bm25_scores_norm, dense_scores_norm, strict=True)
    ]
    extra_fields = [
        {
            "bm25_score": float(bm25_score),
            "bm25_score_norm": float(bm25_score_norm),
            "dense_score": float(dense_score),
            "dense_score_norm": float(dense_score_norm),
        }
        for bm25_score, bm25_score_norm, dense_score, dense_score_norm in zip(
            bm25_scores,
            bm25_scores_norm,
            dense_scores,
            dense_scores_norm,
            strict=True,
        )
    ]
    return rank_windows_by_scores(
        windows=windows,
        scores=fused_scores,
        top_k=top_k,
        score_key="fused_score",
        extra_fields=extra_fields,
    )


def fuse_ranks_rrf(
    *,
    windows: list[dict[str, Any]],
    bm25_scores: list[float],
    dense_scores: list[float],
    rrf_k: int,
    top_k: int,
) -> list[dict[str, Any]]:
    if not windows:
        return []

    bm25_order = sorted(range(len(windows)), key=lambda idx: bm25_scores[idx], reverse=True)
    dense_order = sorted(range(len(windows)), key=lambda idx: dense_scores[idx], reverse=True)
    bm25_ranks = {idx: rank for rank, idx in enumerate(bm25_order, start=1)}
    dense_ranks = {idx: rank for rank, idx in enumerate(dense_order, start=1)}

    fused_scores = [
        (1.0 / (rrf_k + bm25_ranks[idx])) + (1.0 / (rrf_k + dense_ranks[idx]))
        for idx in range(len(windows))
    ]
    extra_fields = [
        {
            "bm25_score": float(bm25_scores[idx]),
            "dense_score": float(dense_scores[idx]),
            "bm25_rank": int(bm25_ranks[idx]),
            "dense_rank": int(dense_ranks[idx]),
        }
        for idx in range(len(windows))
    ]
    return rank_windows_by_scores(
        windows=windows,
        scores=fused_scores,
        top_k=top_k,
        score_key="fused_score",
        extra_fields=extra_fields,
    )


def rerank_bm25_candidates_with_dense(
    *,
    windows: list[dict[str, Any]],
    bm25_scores: list[float],
    dense_scores: list[float],
    top_n: int,
    top_k: int,
) -> list[dict[str, Any]]:
    if not windows:
        return []

    bm25_order = sorted(range(len(windows)), key=lambda idx: bm25_scores[idx], reverse=True)[:top_n]
    candidate_windows = [windows[idx] for idx in bm25_order]
    candidate_dense_scores = [float(dense_scores[idx]) for idx in bm25_order]
    extra_fields = [
        {
            "bm25_score": float(bm25_scores[idx]),
            "dense_score": float(dense_scores[idx]),
            "bm25_rank": rank + 1,
        }
        for rank, idx in enumerate(bm25_order)
    ]
    return rank_windows_by_scores(
        windows=candidate_windows,
        scores=candidate_dense_scores,
        top_k=top_k,
        score_key="dense_score",
        extra_fields=extra_fields,
    )


def score_window_visual_relevance(
    *,
    query_embedding: np.ndarray,
    frame_timestamps: np.ndarray,
    frame_embeddings: np.ndarray,
    start_time: float,
    end_time: float,
    pool: str = "max",
) -> float:
    if frame_timestamps.size == 0 or frame_embeddings.size == 0:
        return 0.0

    mask = (frame_timestamps >= start_time) & (frame_timestamps <= end_time)
    if not np.any(mask):
        center = (start_time + end_time) / 2.0
        nearest_idx = int(np.argmin(np.abs(frame_timestamps - center)))
        mask = np.zeros_like(frame_timestamps, dtype=bool)
        mask[nearest_idx] = True

    selected = frame_embeddings[mask]
    similarities = selected @ query_embedding
    if similarities.size == 0:
        return 0.0
    if pool == "mean":
        return float(np.mean(similarities))
    return float(np.max(similarities))


def rerank_windows_with_visual(
    *,
    windows: list[dict[str, Any]],
    query_embedding: np.ndarray,
    frame_timestamps: np.ndarray,
    frame_embeddings: np.ndarray,
    text_weight: float,
    visual_weight: float,
    pool: str = "max",
) -> list[dict[str, Any]]:
    if not windows:
        return []

    text_scores = [float(window.get("score", 0.0)) for window in windows]
    visual_scores = [
        score_window_visual_relevance(
            query_embedding=query_embedding,
            frame_timestamps=frame_timestamps,
            frame_embeddings=frame_embeddings,
            start_time=float(window["start_time"]),
            end_time=float(window["end_time"]),
            pool=pool,
        )
        for window in windows
    ]
    text_scores_norm = _minmax_normalize(text_scores)
    visual_scores_norm = _minmax_normalize(visual_scores)

    reranked: list[dict[str, Any]] = []
    for window, text_score, visual_score, text_score_norm, visual_score_norm in zip(
        windows,
        text_scores,
        visual_scores,
        text_scores_norm,
        visual_scores_norm,
        strict=True,
    ):
        fused_score = (text_weight * text_score_norm) + (visual_weight * visual_score_norm)
        reranked.append(
            {
                **window,
                "text_score": text_score,
                "text_score_norm": text_score_norm,
                "visual_score": visual_score,
                "visual_score_norm": visual_score_norm,
                "score": fused_score,
            }
        )

    reranked.sort(key=lambda row: float(row["score"]), reverse=True)
    for rank, window in enumerate(reranked, start=1):
        window["rank"] = rank
    return reranked
