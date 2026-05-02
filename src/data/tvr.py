from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..utils import read_jsonl


OFFICIAL_QUERY_FILES: dict[str, list[str]] = {
    "train": ["tvr_train_release.jsonl", "train.jsonl", "tvr_train.jsonl"],
    "val": ["tvr_val_release.jsonl", "val.jsonl", "tvr_val.jsonl", "dev.jsonl"],
    "test_public": ["tvr_test_public_release.jsonl", "tvr_test_release.jsonl", "test_public.jsonl"],
    "test": ["tvr_test_public_release.jsonl", "tvr_test_release.jsonl", "test.jsonl"],
}
OFFICIAL_SUBTITLE_FILE = "tvqa_preprocessed_subtitles.jsonl"
OFFICIAL_VIDEO_DURATION_FILE = "tvr_video2dur_idx.json"


def resolve_query_path(split: str, *, data_dir: str = "data/tvr", query_path: str = "") -> Path:
    if query_path:
        path = Path(query_path)
        if not path.exists():
            raise FileNotFoundError(f"Query file does not exist: {path}")
        return path

    candidates = OFFICIAL_QUERY_FILES.get(split, [f"{split}.jsonl"])
    for candidate in candidates:
        path = Path(data_dir) / candidate
        if path.exists():
            return path
        recursive_matches = sorted(Path(data_dir).rglob(candidate))
        if recursive_matches:
            return recursive_matches[0]

    existing_files = sorted(str(path.relative_to(data_dir)) for path in Path(data_dir).rglob("*.jsonl")) if Path(data_dir).exists() else []
    existing_files_display = ", ".join(existing_files[:20]) if existing_files else "none"

    raise FileNotFoundError(
        f"Could not resolve split '{split}' under {data_dir}. "
        f"Tried filenames: {', '.join(candidates)}. "
        f"Expected TVR split files under data/tvr/, for example data/tvr/tvr_val_release.jsonl. "
        f"Existing JSONL files under {data_dir}: {existing_files_display}"
    )


def _normalize_query_row(row: dict[str, Any], *, split: str) -> dict[str, Any]:
    ts = row.get("ts")
    gold_start_time = None
    gold_end_time = None
    if isinstance(ts, list) and len(ts) == 2:
        gold_start_time = float(ts[0])
        gold_end_time = float(ts[1])

    desc_id = row.get("desc_id")
    query_id = str(desc_id) if desc_id is not None else str(row.get("query_id", ""))
    return {
        "query_id": query_id,
        "desc_id": desc_id,
        "vid_name": str(row.get("vid_name", "")),
        "duration": float(row["duration"]) if row.get("duration") is not None else None,
        "query": str(row.get("desc", row.get("query", ""))),
        "query_type": str(row.get("type", "")),
        "gold_start_time": gold_start_time,
        "gold_end_time": gold_end_time,
        "split": split,
        "raw": row,
    }


def load_tvr_queries(
    split: str = "train",
    *,
    data_dir: str = "data/tvr",
    query_path: str = "",
) -> list[dict[str, Any]]:
    path = resolve_query_path(split, data_dir=data_dir, query_path=query_path)
    return [_normalize_query_row(row, split=split) for row in read_jsonl(path)]


def _normalize_segment(vid_name: str, segment: dict[str, Any], index: int) -> dict[str, Any]:
    text = str(segment.get("text", "")).strip()
    start_time = float(segment.get("start", segment.get("start_time", 0.0)))
    end_time = float(segment.get("end", segment.get("end_time", start_time)))
    return {
        "vid_name": vid_name,
        "segment_id": f"{vid_name}:{index:05d}",
        "start_time": start_time,
        "end_time": end_time,
        "text": text,
    }


def _load_segments_from_jsonl_file(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if not rows:
        return []

    first_row = rows[0]
    if "sub" in first_row or "segments" in first_row:
        segments_by_video: list[dict[str, Any]] = []
        for row in rows:
            vid_name = str(row.get("vid_name", row.get("video_id", "")))
            source_segments = row.get("sub", row.get("segments", []))
            if not vid_name or not isinstance(source_segments, list):
                continue
            segments_by_video.extend(
                _normalize_segment(vid_name, segment, idx)
                for idx, segment in enumerate(source_segments)
                if str(segment.get("text", "")).strip()
            )
        return segments_by_video

    normalized: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        vid_name = str(row.get("vid_name", row.get("video_id", "")))
        if not vid_name:
            continue
        normalized.append(
            {
                "vid_name": vid_name,
                "segment_id": str(row.get("segment_id", f"{vid_name}:{idx:05d}")),
                "start_time": float(row.get("start_time", row.get("start", 0.0))),
                "end_time": float(row.get("end_time", row.get("end", 0.0))),
                "text": str(row.get("text", "")).strip(),
            }
        )
    return normalized


def load_subtitle_segments(source_path: str | Path) -> dict[str, list[dict[str, Any]]]:
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"Subtitle source does not exist: {source}")

    segments_by_video: dict[str, list[dict[str, Any]]] = {}
    if source.is_dir():
        for path in sorted(source.glob("*.jsonl")):
            rows = _load_segments_from_jsonl_file(path)
            if rows:
                segments_by_video[path.stem] = rows
        return segments_by_video

    rows = _load_segments_from_jsonl_file(source)
    for row in rows:
        segments_by_video.setdefault(str(row["vid_name"]), []).append(row)
    return segments_by_video


def load_video_durations(path: str | Path) -> dict[str, float]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Duration file does not exist: {input_path}")

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    durations: dict[str, float] = {}
    if isinstance(payload, dict):
        for value in payload.values():
            if not isinstance(value, dict):
                continue
            for vid_name, duration_and_index in value.items():
                if isinstance(duration_and_index, list) and duration_and_index:
                    durations[vid_name] = float(duration_and_index[0])
                elif isinstance(duration_and_index, (int, float)):
                    durations[vid_name] = float(duration_and_index)
    return durations
