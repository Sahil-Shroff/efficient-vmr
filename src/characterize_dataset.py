from __future__ import annotations

import argparse
import statistics
from collections import Counter

from .data import load_raw_video_rows
from .utils import simple_tokenize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    raw_rows = load_raw_video_rows(split=args.split)
    if not raw_rows:
        print("No records found.")
        return

    transcript_token_lengths = []
    questions_per_video = []
    qa_type_counts: Counter[str] = Counter()

    for row in raw_rows:
        transcript = row.get("caption", row.get("transcript", "")) or ""
        transcript_token_lengths.append(len(simple_tokenize(transcript)))

        reasoning = row.get("reasoning_qa", [])
        captions = row.get("captions_qa", [])
        reasoning_count = len(reasoning) if isinstance(reasoning, list) else 0
        captions_count = len(captions) if isinstance(captions, list) else 0

        qa_type_counts["reasoning_qa"] += reasoning_count
        qa_type_counts["captions_qa"] += captions_count
        questions_per_video.append(reasoning_count + captions_count)

    print(f"Split: {args.split}")
    print(f"Raw video rows: {len(raw_rows)}")
    print(f"Avg transcript tokens: {statistics.mean(transcript_token_lengths):.2f}")
    print(f"Median transcript tokens: {statistics.median(transcript_token_lengths):.2f}")
    print(f"Avg questions per video: {statistics.mean(questions_per_video):.2f}")
    print(f"Median questions per video: {statistics.median(questions_per_video):.2f}")
    print("QA type distribution:")
    for qa_type, count in qa_type_counts.most_common():
        pct = 100.0 * count / max(1, sum(qa_type_counts.values()))
        print(f"  - {qa_type}: {count} ({pct:.2f}%)")


if __name__ == "__main__":
    main()
