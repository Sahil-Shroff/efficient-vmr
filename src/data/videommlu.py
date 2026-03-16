from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset


def _load_local_json(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ["data", "examples", "items", "records"]:
            value = payload.get(key)
            if isinstance(value, list):
                return value

    raise ValueError(f"Unsupported JSON structure in {path}")


def load_raw_video_rows(
    split: str = "Video_MMLU",
    *,
    local_path: str | None = None,
    hf_dataset: str | None = None,
    hf_config: str | None = None,
) -> list[dict[str, Any]]:
    local_path = (local_path if local_path is not None else os.getenv("VIDEOMMLU_LOCAL_PATH", "")).strip()
    hf_dataset = (hf_dataset if hf_dataset is not None else os.getenv("VIDEOMMLU_HF_DATASET", "")).strip()
    hf_config = (hf_config if hf_config is not None else os.getenv("VIDEOMMLU_HF_CONFIG", "")).strip() or None

    if local_path:
        input_path = Path(local_path)
        if not input_path.exists():
            raise FileNotFoundError(f"VIDEOMMLU_LOCAL_PATH does not exist: {input_path}")
        return _load_local_json(input_path)

    if hf_dataset:
        dataset = load_dataset(hf_dataset, hf_config) if hf_config else load_dataset(hf_dataset)
        if split not in dataset:
            raise KeyError(f"Split '{split}' not found. Available splits: {list(dataset.keys())}")
        return [dict(row) for row in dataset[split]]

    raise ValueError("Set either VIDEOMMLU_LOCAL_PATH or VIDEOMMLU_HF_DATASET in the environment.")


def flatten_video_qa_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    flat_rows: list[dict[str, Any]] = []
    for row in rows:
        video_id = row.get("video_id", "")
        transcript = row.get("caption", row.get("transcript", "")) or ""
        for qa_type in ("reasoning_qa", "captions_qa"):
            qa_items = row.get(qa_type, [])
            if not isinstance(qa_items, list):
                continue
            for idx, qa in enumerate(qa_items):
                if not isinstance(qa, dict):
                    continue
                flat_rows.append(
                    {
                        "video_id": video_id,
                        "question_id": f"{video_id}:{qa_type}:{idx}",
                        "question": qa.get("question", ""),
                        "answer": qa.get("answer"),
                        "qa_type": qa_type,
                        "transcript": transcript,
                        "start_time": row.get("start_time"),
                        "end_time": row.get("end_time"),
                        "raw_video": {
                            "video_id": video_id,
                            "caption": transcript,
                        },
                        "raw_qa": qa,
                    }
                )
    return flat_rows


def load_flattened_examples(
    split: str = "Video_MMLU",
    *,
    local_path: str | None = None,
    hf_dataset: str | None = None,
    hf_config: str | None = None,
) -> list[dict[str, Any]]:
    return flatten_video_qa_rows(
        load_raw_video_rows(
            split=split,
            local_path=local_path,
            hf_dataset=hf_dataset,
            hf_config=hf_config,
        )
    )


def extract_unique_video_ids(rows: Iterable[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for row in rows:
        video_id = (row.get("video_id") or "").strip()
        if video_id and video_id not in seen:
            seen.add(video_id)
            ordered.append(video_id)
    return ordered
