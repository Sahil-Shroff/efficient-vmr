from __future__ import annotations

import argparse
from pathlib import Path

from .videommlu import extract_unique_video_ids, load_raw_video_rows
from ..utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract unique Video-MMLU video IDs.")
    parser.add_argument("--split", default="Video_MMLU")
    parser.add_argument("--output", default="data/video_ids.txt")
    args = parser.parse_args()

    rows = load_raw_video_rows(split=args.split)
    video_ids = extract_unique_video_ids(rows)

    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    output_path.write_text("\n".join(video_ids) + "\n", encoding="utf-8")

    print(f"Wrote {len(video_ids)} unique video_ids to {output_path}")


if __name__ == "__main__":
    main()
