from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - handled at runtime
    SentenceTransformer = None  # type: ignore[assignment]


def _require_sentence_transformers() -> None:
    if SentenceTransformer is None:
        raise ImportError(
            "sentence-transformers is not installed. "
            "Install requirements with `pip install -r requirements.txt`."
        )


@dataclass
class DenseTextRetriever:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str | None = None
    batch_size: int = 64
    normalize_embeddings: bool = True
    cache_folder: str | None = None

    def __post_init__(self) -> None:
        _require_sentence_transformers()
        self.model = SentenceTransformer(
            self.model_name,
            device=self.device,
            cache_folder=self.cache_folder,
        )

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        if hasattr(self.model, "encode_document"):
            return np.asarray(
                self.model.encode_document(
                    texts,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=self.normalize_embeddings,
                    convert_to_numpy=True,
                ),
                dtype=np.float32,
            )
        return np.asarray(
            self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True,
            ),
            dtype=np.float32,
        )

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        if hasattr(self.model, "encode_query"):
            return np.asarray(
                self.model.encode_query(
                    texts,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=self.normalize_embeddings,
                    convert_to_numpy=True,
                ),
                dtype=np.float32,
            )
        return np.asarray(
            self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True,
            ),
            dtype=np.float32,
        )


def build_dense_index(
    windows: list[dict[str, Any]],
    *,
    retriever: DenseTextRetriever,
) -> np.ndarray:
    texts = [str(window.get("text", "")) for window in windows]
    return retriever.encode_documents(texts)


def retrieve_top_k_windows_dense(
    *,
    query: str,
    windows: list[dict[str, Any]],
    document_embeddings: np.ndarray,
    retriever: DenseTextRetriever,
    top_k: int,
) -> list[dict[str, Any]]:
    if not windows:
        return []

    query_embedding = retriever.encode_queries([query])[0]
    return retrieve_top_k_windows_from_embedding(
        windows=windows,
        document_embeddings=document_embeddings,
        query_embedding=query_embedding,
        top_k=top_k,
    )


def retrieve_top_k_windows_from_embedding(
    *,
    windows: list[dict[str, Any]],
    document_embeddings: np.ndarray,
    query_embedding: np.ndarray,
    top_k: int,
) -> list[dict[str, Any]]:
    if not windows:
        return []

    scores = score_windows_dense(
        document_embeddings=document_embeddings,
        query_embedding=query_embedding,
    )
    ranked = sorted(enumerate(scores.tolist()), key=lambda item: item[1], reverse=True)[:top_k]
    return [
        {
            **windows[idx],
            "score": float(score),
            "rank": rank + 1,
        }
        for rank, (idx, score) in enumerate(ranked)
    ]


def score_windows_dense(
    *,
    document_embeddings: np.ndarray,
    query_embedding: np.ndarray,
) -> np.ndarray:
    if document_embeddings.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return np.asarray(document_embeddings @ query_embedding, dtype=np.float32)
