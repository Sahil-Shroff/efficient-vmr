#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import OFFICIAL_SUBTITLE_FILE, load_subtitle_segments, load_tvr_queries


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect TVR split metadata and subtitle coverage.")
    parser.add_argument("--data-dir", default="data/tvr")
    parser.add_argument("--split", default="val")
    parser.add_argument("--query-path", default="")
    parser.add_argument("--subtitles-path", default="")
    args = parser.parse_args()

    try:
        queries = load_tvr_queries(split=args.split, data_dir=args.data_dir, query_path=args.query_path)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"{exc}\n\n"
            "Place the official TVR files under data/tvr/ as documented in data/README.md, "
            "or pass --query-path directly to a split JSONL file."
        )
    durations = [query["duration"] for query in queries if query.get("duration") is not None]
    summary = {
        "split": args.split,
        "num_queries": len(queries),
        "num_unique_videos": len({query["vid_name"] for query in queries}),
        "query_type_counts": {
            query_type: sum(1 for query in queries if query.get("query_type") == query_type)
            for query_type in ("v", "t", "vt")
        },
        "duration_min": min(durations) if durations else None,
        "duration_max": max(durations) if durations else None,
        "duration_mean": (sum(durations) / len(durations)) if durations else None,
    }

    subtitles_path = args.subtitles_path or str(Path(args.data_dir) / OFFICIAL_SUBTITLE_FILE)
    if Path(subtitles_path).exists():
        subtitle_segments = load_subtitle_segments(subtitles_path)
        summary["subtitle_videos"] = len(subtitle_segments)
        summary["subtitle_coverage_on_split"] = (
            sum(1 for query in queries if query["vid_name"] in subtitle_segments) / len(queries)
            if queries
            else None
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
