from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..data import load_flattened_examples
from ..eval.moment_retrieval import _filter_examples_by_video_ids, _load_requested_video_ids
from ..retrieval import build_bm25_index, retrieve_top_k_windows
from ..utils import ensure_dir, read_jsonl, write_jsonl


DEFAULT_ANNOTATIONS_PATH = "data/annotations/vmr_gold_annotations.jsonl"


def _default_data_path(candidates: list[str]) -> str:
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return candidates[-1]


@dataclass(frozen=True)
class AnnotationWorkspaceConfig:
    split: str = "Video_MMLU"
    dataset_path: str = _default_data_path(
        [
            "data/subtitles/vmr10/raw_video_rows.jsonl",
            "data/raw_video_rows.jsonl",
        ]
    )
    windows_dir: str = _default_data_path(
        [
            "data/subtitles/vmr10/windows",
            "data/subtitles/windows",
        ]
    )
    parsed_dir: str = _default_data_path(
        [
            "data/subtitles/vmr10/parsed",
            "data/subtitles/parsed",
        ]
    )
    annotations_path: str = DEFAULT_ANNOTATIONS_PATH
    video_ids_path: str = ""
    top_k: int = 3


def load_annotations_map(path: str | Path) -> dict[str, dict[str, Any]]:
    annotations: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        question_id = str(row.get("question_id", "")).strip()
        if question_id:
            annotations[question_id] = row
    return annotations


def parse_gold_timestamps(raw_text: str) -> list[dict[str, float]]:
    if not raw_text.strip():
        return []

    payload = json.loads(raw_text)
    if not isinstance(payload, list):
        raise ValueError("gold_timestamps must be a JSON list.")

    normalized_spans: list[dict[str, float]] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Span {idx + 1} must be an object with start/end.")
        if "start" not in item or "end" not in item:
            raise ValueError(f"Span {idx + 1} must include start and end.")

        start = float(item["start"])
        end = float(item["end"])
        if end < start:
            raise ValueError(f"Span {idx + 1} has end < start.")

        normalized_spans.append({"start": start, "end": end})
    return normalized_spans


class AnnotationWorkspace:
    def __init__(self, config: AnnotationWorkspaceConfig) -> None:
        self.config = config
        video_ids = _load_requested_video_ids(config.video_ids_path or None)
        examples = load_flattened_examples(split=config.split, local_path=config.dataset_path)
        self.examples = _filter_examples_by_video_ids(examples, video_ids=video_ids, max_videos=0)
        self.examples_by_question_id = {
            str(example["question_id"]): example for example in self.examples if example.get("question_id")
        }
        self.annotations_path = Path(config.annotations_path)
        self.annotations_by_question_id = load_annotations_map(self.annotations_path)
        self._windows_by_video = self._load_jsonl_dir(config.windows_dir)
        self._segments_by_video = self._load_jsonl_dir(config.parsed_dir)
        self._retrieval_cache: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
        self._bm25_cache: dict[str, tuple[list[dict[str, Any]], Any]] = {}

    @staticmethod
    def _load_jsonl_dir(path: str) -> dict[str, list[dict[str, Any]]]:
        base = Path(path)
        if not base.exists():
            return {}

        rows_by_stem: dict[str, list[dict[str, Any]]] = {}
        for file_path in sorted(base.glob("*.jsonl")):
            rows_by_stem[file_path.stem] = read_jsonl(file_path)
        return rows_by_stem

    def list_video_ids(self) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for example in self.examples:
            video_id = str(example.get("video_id", "")).strip()
            if video_id and video_id not in seen:
                seen.add(video_id)
                ordered.append(video_id)
        return ordered

    def get_example(self, question_id: str) -> dict[str, Any] | None:
        return self.examples_by_question_id.get(question_id)

    def get_annotation(self, question_id: str) -> dict[str, Any] | None:
        return self.annotations_by_question_id.get(question_id)

    def get_windows(self, video_id: str) -> list[dict[str, Any]]:
        return self._windows_by_video.get(video_id, [])

    def get_segments(self, video_id: str) -> list[dict[str, Any]]:
        return self._segments_by_video.get(video_id, [])

    def get_retrieval_candidates(self, question_id: str, *, top_k: int | None = None) -> list[dict[str, Any]]:
        example = self.get_example(question_id)
        if example is None:
            return []

        video_id = str(example.get("video_id", ""))
        question = str(example.get("question", ""))
        windows = self.get_windows(video_id)
        if not windows:
            return []

        limit = top_k or self.config.top_k
        cache_key = (question_id, video_id, limit)
        if cache_key in self._retrieval_cache:
            return self._retrieval_cache[cache_key]

        if video_id not in self._bm25_cache:
            self._bm25_cache[video_id] = (windows, build_bm25_index(windows)[0])

        cached_windows, bm25 = self._bm25_cache[video_id]
        retrieved = retrieve_top_k_windows(
            query=question,
            windows=cached_windows,
            bm25=bm25,
            top_k=limit,
        )
        self._retrieval_cache[cache_key] = retrieved
        return retrieved

    def save_annotation(self, row: dict[str, Any]) -> None:
        question_id = str(row.get("question_id", "")).strip()
        if not question_id:
            raise ValueError("question_id is required.")

        self.annotations_by_question_id[question_id] = row
        ensure_dir(self.annotations_path.parent)

        ordered_rows: list[dict[str, Any]] = []
        for example in self.examples:
            example_question_id = str(example.get("question_id", "")).strip()
            if example_question_id in self.annotations_by_question_id:
                ordered_rows.append(self.annotations_by_question_id[example_question_id])

        extra_question_ids = sorted(set(self.annotations_by_question_id) - {row["question_id"] for row in ordered_rows})
        for extra_question_id in extra_question_ids:
            ordered_rows.append(self.annotations_by_question_id[extra_question_id])

        write_jsonl(self.annotations_path, ordered_rows)
