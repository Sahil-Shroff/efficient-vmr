#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import site
import sys
from typing import Any

import numpy as np
from tqdm import tqdm


def _configure_wheel_cuda_lib_path() -> None:
    lib_dirs: list[str] = []
    for root in (site.getusersitepackages(), *site.getsitepackages()):
        if not root:
            continue
        base = Path(root)
        for path in (
            base / "nvidia" / "nvjitlink" / "lib",
            base / "nvidia" / "cusparse" / "lib",
            base / "nvidia" / "cublas" / "lib",
            base / "nvidia" / "cudnn" / "lib",
            base / "nvidia" / "cuda_runtime" / "lib",
            base / "nvidia" / "cuda_nvrtc" / "lib",
        ):
            if path.is_dir():
                lib_dirs.append(str(path))
    if lib_dirs:
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(lib_dirs + ([existing] if existing else []))


_configure_wheel_cuda_lib_path()

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import (  # noqa: E402
    OFFICIAL_SUBTITLE_FILE,
    OFFICIAL_VIDEO_DURATION_FILE,
    load_subtitle_segments,
    load_tvr_queries,
)
from src.retrieval.dense import DenseTextRetriever  # noqa: E402
from src.utils import Timer, ensure_dir, write_json  # noqa: E402


def _resolve_split_name(split: str) -> str:
    if split == "test":
        return "test_public"
    return split


def _load_video2idx(duration_path: Path, split: str) -> dict[str, int]:
    payload = json.loads(duration_path.read_text(encoding="utf-8"))
    split_payload = payload[_resolve_split_name(split)]
    return {
        vid_name: int(duration_and_idx[1])
        for vid_name, duration_and_idx in split_payload.items()
    }


def _build_video_documents(
    *,
    video2idx: dict[str, int],
    segments_by_video: dict[str, list[dict[str, Any]]],
) -> tuple[list[str], list[str]]:
    ordered_videos = [vid_name for vid_name, _ in sorted(video2idx.items(), key=lambda item: item[1])]
    documents: list[str] = []
    for vid_name in ordered_videos:
        segments = segments_by_video.get(vid_name, [])
        text = " ".join(
            str(segment.get("text", "")).strip()
            for segment in segments
            if str(segment.get("text", "")).strip()
        )
        documents.append(text if text else vid_name.replace("_", " "))
    return ordered_videos, documents


def _topk_indices(scores: np.ndarray, top_k: int) -> np.ndarray:
    top_k = min(top_k, len(scores))
    if top_k <= 0:
        return np.zeros((0,), dtype=np.int64)
    if top_k == len(scores):
        return np.argsort(scores)[::-1]
    candidate_indices = np.argpartition(scores, -top_k)[-top_k:]
    return candidate_indices[np.argsort(scores[candidate_indices])[::-1]]


def _empty_recall_summary() -> dict[str, dict[str, float | int]]:
    return {str(n): {"hits": 0, "total": 0, "recall": 0.0} for n in (1, 3, 5, 10)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dense video shortlists for XML TVR inference.")
    parser.add_argument("--data-dir", default="third_party/TVRetrieval/data")
    parser.add_argument("--split", default="val")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--subtitles-path", default="")
    parser.add_argument("--duration-path", default="")
    parser.add_argument("--output-dir", default="outputs/tvr/xml_hierarchical/dense_shortlist")
    parser.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", default="")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cache-folder", default="")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    subtitles_path = Path(args.subtitles_path) if args.subtitles_path else data_dir / OFFICIAL_SUBTITLE_FILE
    duration_path = Path(args.duration_path) if args.duration_path else data_dir / OFFICIAL_VIDEO_DURATION_FILE
    output_dir = ensure_dir(args.output_dir)

    queries = load_tvr_queries(split=args.split, data_dir=str(data_dir))
    segments_by_video = load_subtitle_segments(subtitles_path)
    video2idx = _load_video2idx(duration_path, split=args.split)
    ordered_videos, documents = _build_video_documents(video2idx=video2idx, segments_by_video=segments_by_video)

    retriever = DenseTextRetriever(
        model_name=args.model_name,
        device=args.device or ("cuda" if torch.cuda.is_available() else None),
        batch_size=args.batch_size,
        cache_folder=args.cache_folder or None,
    )

    with Timer() as timer:
        document_embeddings = retriever.encode_documents(documents)
    document_encoding_s = timer.elapsed_s

    with Timer() as timer:
        query_embeddings = retriever.encode_queries([str(query["query"]) for query in queries])
    query_encoding_s = timer.elapsed_s

    shortlist_rows = []
    recall_summary = _empty_recall_summary()
    recall_by_type = {query_type: _empty_recall_summary() for query_type in ("v", "t", "vt")}

    with Timer() as timer:
        for query, query_embedding in tqdm(
            zip(queries, query_embeddings, strict=True),
            total=len(queries),
            desc="Dense video shortlist",
        ):
            scores = np.asarray(document_embeddings @ query_embedding, dtype=np.float32)
            top_indices = _topk_indices(scores, args.top_k)
            predictions = [
                [video2idx[ordered_videos[idx]], 0, 0, float(scores[idx])]
                for idx in top_indices.tolist()
            ]
            shortlist_rows.append(
                {
                    "desc_id": query["desc_id"],
                    "desc": query["query"],
                    "predictions": predictions,
                }
            )

            ranked_videos = [ordered_videos[idx] for idx in top_indices.tolist()]
            gold_vid = str(query["vid_name"])
            query_type = str(query["query_type"])
            for n in (1, 3, 5, 10):
                bucket = recall_summary[str(n)]
                bucket["total"] += 1
                if gold_vid in ranked_videos[:n]:
                    bucket["hits"] += 1

                typed_bucket = recall_by_type[query_type][str(n)]
                typed_bucket["total"] += 1
                if gold_vid in ranked_videos[:n]:
                    typed_bucket["hits"] += 1
    retrieval_s = timer.elapsed_s

    for bucket in recall_summary.values():
        if bucket["total"]:
            bucket["recall"] = bucket["hits"] / bucket["total"]
    for type_buckets in recall_by_type.values():
        for bucket in type_buckets.values():
            if bucket["total"]:
                bucket["recall"] = bucket["hits"] / bucket["total"]

    shortlist_payload = {
        "video2idx": video2idx,
        "VR": shortlist_rows,
    }
    shortlist_path = output_dir / f"predictions_{args.split}_top{args.top_k}.json"
    write_json(shortlist_path, shortlist_payload)

    summary = {
        "split": args.split,
        "model_name": args.model_name,
        "device": args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        "num_queries": len(queries),
        "num_videos": len(ordered_videos),
        "top_k": args.top_k,
        "document_encoding_s": document_encoding_s,
        "query_encoding_s": query_encoding_s,
        "retrieval_scoring_s": retrieval_s,
        "video_recall_at_n": recall_summary,
        "video_recall_at_n_by_type": recall_by_type,
        "shortlist_path": str(shortlist_path),
    }
    summary_path = output_dir / f"summary_{args.split}_top{args.top_k}.json"
    write_json(summary_path, summary)
    print(summary_path)


if __name__ == "__main__":
    main()
