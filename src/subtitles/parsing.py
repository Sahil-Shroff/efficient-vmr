from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ..utils import ensure_dir, normalize_text, normalize_whitespace, read_jsonl_records, write_jsonl


def normalize_subtitle_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    vid_name = str(row.get("vid_name", row.get("video_id", ""))).strip()
    raw_segments = row.get("sub", row.get("segments", []))
    if not vid_name or not isinstance(raw_segments, list):
        return []

    segments: list[dict[str, Any]] = []
    for raw_segment in raw_segments:
        if not isinstance(raw_segment, dict):
            continue

        text = normalize_whitespace(str(raw_segment.get("text", "")))
        if not text:
            continue

        start_time = float(raw_segment.get("start", raw_segment.get("start_time", 0.0)))
        end_time = float(raw_segment.get("end", raw_segment.get("end_time", start_time)))
        if segments and normalize_text(segments[-1]["text"]) == normalize_text(text):
            segments[-1]["end_time"] = end_time
            continue

        segments.append(
            {
                "vid_name": vid_name,
                "segment_id": f"{vid_name}:{len(segments):05d}",
                "start_time": start_time,
                "end_time": end_time,
                "text": text,
            }
        )
    return segments


def parse_subtitles_jsonl_to_dir(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    force: bool = False,
    limit: int = 0,
) -> Path:
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Subtitle JSONL file does not exist: {input_file}")

    output_path = ensure_dir(output_dir)
    for index, row in enumerate(tqdm(read_jsonl_records(input_file), desc="Normalizing subtitles")):
        if limit > 0 and index >= limit:
            break
        vid_name = str(row.get("vid_name", row.get("video_id", ""))).strip()
        if not vid_name:
            continue

        clip_output_path = output_path / f"{vid_name}.jsonl"
        if clip_output_path.exists() and not force:
            continue

        write_jsonl(clip_output_path, normalize_subtitle_row(row))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize official TVR subtitle JSONL into per-clip segment files.")
    parser.add_argument("--input-path", default="data/tvr/tvqa_preprocessed_subtitles.jsonl")
    parser.add_argument("--output-dir", default="data/tvr/processed/subtitles")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    output_dir = parse_subtitles_jsonl_to_dir(
        input_path=args.input_path,
        output_dir=args.output_dir,
        force=args.force,
        limit=args.limit,
    )
    print(f"Subtitle segments written under {output_dir}")


if __name__ == "__main__":
    main()
