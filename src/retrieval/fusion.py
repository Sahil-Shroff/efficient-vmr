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
