from __future__ import annotations

import argparse
from collections import Counter

from .load_videommlu import flatten_videommlu, load_raw_videommlu


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-examples", type=int, default=3)
    args = parser.parse_args()

    raw_rows = load_raw_videommlu(split=args.split)
    flat_rows = flatten_videommlu(raw_rows)

    raw_video_ids = [row.get("video_id", "") for row in raw_rows]
    flat_video_ids = [row.get("video_id", "") for row in flat_rows]
    per_video_counts = Counter(video_id or "unknown" for video_id in flat_video_ids)

    print(f"Split: {args.split}")
    print(f"Raw video rows: {len(raw_rows)}")
    print(f"Flattened QA rows: {len(flat_rows)}")
    print(f"Unique video_ids in raw rows: {len(set(raw_video_ids))}")
    print(f"Unique video_ids in flattened rows: {len(set(flat_video_ids))}")
    print(
        "Avg QA rows per video: "
        f"{len(flat_rows) / max(1, len(set(flat_video_ids))):.2f}"
    )

    print("\nFirst few raw examples:")
    for idx, row in enumerate(raw_rows[: args.num_examples]):
        reasoning = row.get("reasoning_qa", [])
        captions = row.get("captions_qa", [])
        print(
            f"[raw {idx}] video_id={row.get('video_id')} "
            f"reasoning_qa={len(reasoning) if isinstance(reasoning, list) else 0} "
            f"captions_qa={len(captions) if isinstance(captions, list) else 0}"
        )

    print("\nFirst few flattened examples:")
    for idx, row in enumerate(flat_rows[: args.num_examples]):
        print(
            f"[flat {idx}] video_id={row.get('video_id')} "
            f"question_id={row.get('question_id')} "
            f"subject={row.get('subject')} "
            f"question={row.get('question')!r}"
        )

    print("\nSample per-video expansion counts:")
    for video_id, count in per_video_counts.most_common(args.num_examples):
        print(f"  {video_id}: {count}")


if __name__ == "__main__":
    main()
