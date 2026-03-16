from __future__ import annotations

import argparse
import re
from pathlib import Path

from tqdm import tqdm

from ..utils import ensure_dir, normalize_text, read_jsonl, strip_markup, write_jsonl


TIMESTAMP_RE = re.compile(
    r"(?P<start>\d{2}:\d{2}(?::\d{2})?\.\d{3})\s+-->\s+(?P<end>\d{2}:\d{2}(?::\d{2})?\.\d{3})"
)


def _timestamp_to_seconds(raw_value: str) -> float:
    parts = raw_value.split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        hours = 0
    else:
        hours, minutes, seconds = parts
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def parse_vtt_text(text: str, *, video_id: str) -> list[dict[str, object]]:
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n"))
    segments: list[dict[str, object]] = []

    for block in blocks:
        lines = [line.strip("\ufeff") for line in block.splitlines() if line.strip()]
        if not lines:
            continue

        timestamp_index = next((idx for idx, line in enumerate(lines) if "-->" in line), None)
        if timestamp_index is None:
            continue

        match = TIMESTAMP_RE.search(lines[timestamp_index])
        if not match:
            continue

        start_time = _timestamp_to_seconds(match.group("start"))
        end_time = _timestamp_to_seconds(match.group("end"))
        text_lines = [strip_markup(line) for line in lines[timestamp_index + 1 :]]
        cleaned_text = " ".join(line for line in text_lines if line).strip()
        if not cleaned_text:
            continue

        if segments and normalize_text(segments[-1]["text"]) == normalize_text(cleaned_text):
            segments[-1]["end_time"] = end_time
            continue

        segment_id = f"{video_id}:{len(segments):05d}"
        segments.append(
            {
                "video_id": video_id,
                "segment_id": segment_id,
                "start_time": start_time,
                "end_time": end_time,
                "text": cleaned_text,
            }
        )

    return segments


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse raw VTT subtitles into normalized timestamped segments.")
    parser.add_argument("--manifest-path", default="data/subtitles/subtitle_manifest.jsonl")
    parser.add_argument("--output-dir", default="data/subtitles/parsed")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest_rows = read_jsonl(args.manifest_path)
    output_dir = ensure_dir(args.output_dir)

    for row in tqdm(manifest_rows, desc="Parsing subtitles"):
        if row.get("status") not in {"success_human", "success_auto"}:
            continue

        subtitle_path = row.get("subtitle_path")
        video_id = row.get("video_id")
        if not subtitle_path or not video_id:
            continue

        output_path = output_dir / f"{video_id}.jsonl"
        if output_path.exists() and not args.force:
            continue

        vtt_text = Path(subtitle_path).read_text(encoding="utf-8")
        segments = parse_vtt_text(vtt_text, video_id=video_id)
        write_jsonl(output_path, segments)

    print(f"Parsed subtitle segments written under {output_dir}")


if __name__ == "__main__":
    main()
