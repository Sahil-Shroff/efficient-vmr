from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from ..utils import ensure_dir, read_jsonl, write_jsonl


def build_windows_from_segments(
    segments: list[dict[str, object]],
    *,
    window_size_sec: float,
    stride_sec: float,
) -> list[dict[str, object]]:
    if not segments:
        return []

    video_id = str(segments[0]["video_id"])
    max_end_time = max(float(segment["end_time"]) for segment in segments)
    windows: list[dict[str, object]] = []
    window_start = 0.0

    while window_start < max_end_time:
        window_end = min(max_end_time, window_start + window_size_sec)
        overlapping = [
            segment
            for segment in segments
            if float(segment["start_time"]) < window_end and float(segment["end_time"]) > window_start
        ]
        if overlapping:
            window_id = f"{video_id}:{int(window_start * 1000):09d}:{int(window_end * 1000):09d}"
            window_text = " ".join(str(segment["text"]) for segment in overlapping).strip()
            windows.append(
                {
                    "video_id": video_id,
                    "window_id": window_id,
                    "start_time": round(window_start, 3),
                    "end_time": round(window_end, 3),
                    "text": window_text,
                }
            )
        if window_end >= max_end_time:
            break
        window_start += stride_sec

    return windows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fixed temporal windows from parsed subtitle segments.")
    parser.add_argument("--parsed-dir", default="data/subtitles/parsed")
    parser.add_argument("--output-dir", default="data/subtitles/windows")
    parser.add_argument("--window-size-sec", type=float, default=12.0)
    parser.add_argument("--stride-sec", type=float, default=6.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    parsed_dir = Path(args.parsed_dir)
    output_dir = ensure_dir(args.output_dir)
    parsed_files = sorted(parsed_dir.glob("*.jsonl"))

    for parsed_file in tqdm(parsed_files, desc="Building windows"):
        output_path = output_dir / parsed_file.name
        if output_path.exists() and not args.force:
            continue

        segments = read_jsonl(parsed_file)
        windows = build_windows_from_segments(
            segments,
            window_size_sec=args.window_size_sec,
            stride_sec=args.stride_sec,
        )
        write_jsonl(output_path, windows)

    print(f"Window files written under {output_dir}")


if __name__ == "__main__":
    main()
