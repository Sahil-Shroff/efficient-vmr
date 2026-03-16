from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

from datasets import load_dataset


COMMON_TRANSCRIPT_KEYS = [
    "transcript",
    "subtitle",
    "subtitles",
    "captions",
    "caption",
    "asr",
]


COMMON_QUESTION_KEYS = ["question", "query"]
COMMON_CHOICES_KEYS = ["choices", "options", "candidates"]
COMMON_ANSWER_KEYS = ["answer", "label", "target"]
COMMON_VIDEO_ID_KEYS = ["video_id", "video", "youtube_id"]
COMMON_SUBJECT_KEYS = ["subject", "domain", "topic"]


def _first_present(d: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    return default


def canonicalize_record(row: Dict[str, Any]) -> Dict[str, Any]:
    transcript = _first_present(row, COMMON_TRANSCRIPT_KEYS, "")
    question = _first_present(row, COMMON_QUESTION_KEYS, "")
    choices = _first_present(row, COMMON_CHOICES_KEYS, [])
    answer = _first_present(row, COMMON_ANSWER_KEYS, None)
    video_id = _first_present(row, COMMON_VIDEO_ID_KEYS, "")
    subject = _first_present(row, COMMON_SUBJECT_KEYS, "")

    if isinstance(choices, dict):
        choices = [choices[k] for k in sorted(choices.keys())]

    return {
        "video_id": video_id,
        "question_id": row.get("question_id", row.get("qid", "")),
        "question": question,
        "choices": choices,
        "answer": answer,
        "subject": subject,
        "transcript": transcript,
        "start_time": row.get("start_time"),
        "end_time": row.get("end_time"),
        "raw": row,
    }


def _flatten_videommlu_nested_row(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    qa_groups = [
        ("reasoning_qa", row.get("reasoning_qa", [])),
        ("captions_qa", row.get("captions_qa", [])),
    ]
    flat_rows: List[Dict[str, Any]] = []

    for qa_type, qa_items in qa_groups:
        if not isinstance(qa_items, list):
            continue

        for idx, qa in enumerate(qa_items):
            if not isinstance(qa, dict):
                continue

            flat_rows.append(
                {
                    "video_id": row.get("video_id", ""),
                    "question_id": f"{row.get('video_id', 'unknown')}:{qa_type}:{idx}",
                    "question": qa.get("question", ""),
                    # Video-MMLU provides free-form QA here, not multiple choice.
                    "choices": [],
                    "answer": qa.get("answer"),
                    "subject": qa_type,
                    "transcript": row.get("caption", ""),
                    "start_time": row.get("start_time"),
                    "end_time": row.get("end_time"),
                    "qa_type": qa_type,
                    "raw": {
                        "video_id": row.get("video_id", ""),
                        "caption": row.get("caption", ""),
                        "qa": qa,
                    },
                }
            )

    return flat_rows


def is_videommlu_nested_row(row: Dict[str, Any]) -> bool:
    return "caption" in row and (
        isinstance(row.get("reasoning_qa"), list)
        or isinstance(row.get("captions_qa"), list)
    )


def flatten_videommlu(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flat_rows: List[Dict[str, Any]] = []
    for row in rows:
        if is_videommlu_nested_row(row):
            flat_rows.extend(_flatten_videommlu_nested_row(row))
        else:
            flat_rows.append(canonicalize_record(row))
    return flat_rows


def _load_local_json(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ["data", "examples", "items", "records"]:
            if key in obj and isinstance(obj[key], list):
                return obj[key]
    raise ValueError(f"Unsupported JSON structure in {path}")


def load_raw_videommlu(split: str = "train") -> List[Dict[str, Any]]:
    local_path = os.getenv("VIDEOMMLU_LOCAL_PATH", "").strip()
    hf_dataset = os.getenv("VIDEOMMLU_HF_DATASET", "").strip()
    hf_config = os.getenv("VIDEOMMLU_HF_CONFIG", "").strip() or None

    if local_path:
        path = Path(local_path)
        if not path.exists():
            raise FileNotFoundError(f"VIDEOMMLU_LOCAL_PATH does not exist: {path}")
        return _load_local_json(path)

    if hf_dataset:
        if hf_config:
            ds = load_dataset(hf_dataset, hf_config)
        else:
            ds = load_dataset(hf_dataset)
        if split not in ds:
            available = list(ds.keys())
            raise KeyError(f"Split '{split}' not found. Available splits: {available}")
        return [dict(r) for r in ds[split]]

    raise ValueError(
        "Set either VIDEOMMLU_LOCAL_PATH or VIDEOMMLU_HF_DATASET in .env"
    )


def load_records(split: str = "train") -> List[Dict[str, Any]]:
    return flatten_videommlu(load_raw_videommlu(split=split))
