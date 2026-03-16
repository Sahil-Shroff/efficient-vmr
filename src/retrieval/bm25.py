from __future__ import annotations

from rank_bm25 import BM25Okapi

from ..utils import simple_tokenize


def build_bm25_index(windows: list[dict[str, object]]) -> tuple[BM25Okapi, list[list[str]]]:
    tokenized_windows = [simple_tokenize(str(window.get("text", ""))) for window in windows]
    return BM25Okapi(tokenized_windows), tokenized_windows


def retrieve_top_k_windows(
    *,
    query: str,
    windows: list[dict[str, object]],
    bm25: BM25Okapi,
    top_k: int,
) -> list[dict[str, object]]:
    if not windows:
        return []

    scores = bm25.get_scores(simple_tokenize(query))
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]
    return [
        {
            **windows[idx],
            "score": float(score),
            "rank": rank + 1,
        }
        for rank, (idx, score) in enumerate(ranked)
    ]
