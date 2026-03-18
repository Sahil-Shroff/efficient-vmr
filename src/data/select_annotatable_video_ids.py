from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from .videommlu import extract_unique_video_ids, load_raw_video_rows
from ..utils import ensure_dir


def _load_available_video_ids(parsed_dir: str, windows_dir: str) -> set[str]:
    available_video_ids: set[str] = set()

    for directory in (parsed_dir, windows_dir):
        base = Path(directory)
        if not base.exists():
            continue
        for path in sorted(base.glob("*.jsonl")):
            available_video_ids.add(path.stem)

    return available_video_ids


def _is_watchable_on_youtube(video_id: str) -> bool:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "yt_dlp",
            "--simulate",
            "--skip-download",
            "--quiet",
            "--no-warnings",
            f"https://www.youtube.com/watch?v={video_id}",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write a deterministic subset of Video-MMLU video_ids that have timed subtitle artifacts."
    )
    parser.add_argument("--split", default="Video_MMLU")
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--parsed-dir", default="data/subtitles/parsed")
    parser.add_argument("--windows-dir", default="data/subtitles/windows")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output", default="data/annotations/annotatable_video_ids.txt")
    parser.add_argument(
        "--require-youtube-availability",
        action="store_true",
        help="Validate that each selected video is currently watchable on YouTube via yt-dlp.",
    )
    args = parser.parse_args()

    rows = load_raw_video_rows(split=args.split, local_path=args.dataset_path or None)
    ordered_video_ids = extract_unique_video_ids(rows)
    available_video_ids = _load_available_video_ids(args.parsed_dir, args.windows_dir)

    selected_video_ids: list[str] = []
    for video_id in ordered_video_ids:
        if video_id not in available_video_ids:
            continue
        if args.require_youtube_availability and not _is_watchable_on_youtube(video_id):
            continue

        selected_video_ids.append(video_id)
        if args.limit > 0 and len(selected_video_ids) >= args.limit:
            break

    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    output_path.write_text("\n".join(selected_video_ids) + ("\n" if selected_video_ids else ""), encoding="utf-8")

    print(
        f"Wrote {len(selected_video_ids)} annotatable video_ids to {output_path} "
        f"(available timed-subtitle videos found: {len(available_video_ids)})"
    )


if __name__ == "__main__":
    main()
