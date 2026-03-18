#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import OFFICIAL_SUBTITLE_FILE
from src.subtitles import build_windows_in_dir, parse_subtitles_jsonl_to_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize TVR subtitles and build retrieval windows.")
    parser.add_argument("--data-dir", default="data/tvr")
    parser.add_argument("--subtitles-path", default="")
    parser.add_argument("--parsed-dir", default="data/tvr/processed/subtitles")
    parser.add_argument("--windows-dir", default="data/tvr/processed/windows")
    parser.add_argument("--window-size-sec", type=float, default=12.0)
    parser.add_argument("--stride-sec", type=float, default=6.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-parse", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    subtitles_path = args.subtitles_path or str(Path(args.data_dir) / OFFICIAL_SUBTITLE_FILE)
    if not args.skip_parse:
        parse_subtitles_jsonl_to_dir(
            input_path=subtitles_path,
            output_dir=args.parsed_dir,
            force=args.force,
            limit=args.limit,
        )

    build_windows_in_dir(
        parsed_dir=args.parsed_dir,
        output_dir=args.windows_dir,
        window_size_sec=args.window_size_sec,
        stride_sec=args.stride_sec,
        force=args.force,
    )

    print(f"Built TVR windows under {args.windows_dir}")


if __name__ == "__main__":
    main()
