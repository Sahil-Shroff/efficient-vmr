from __future__ import annotations

import argparse
import statistics
from collections import Counter

from .load_videommlu import load_records
from .utils import simple_tokenize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    records = load_records(split=args.split)
    if not records:
        print("No records found.")
        return

    transcript_token_lengths = [len(simple_tokenize(r.get("transcript", ""))) for r in records]
    num_choices = [len(r.get("choices", [])) for r in records]
    has_time = [r.get("start_time") is not None and r.get("end_time") is not None for r in records]
    subject_counts = Counter(r.get("subject", "unknown") or "unknown" for r in records)
    video_counts = Counter(r.get("video_id", "") or "unknown" for r in records)

    print(f"Split: {args.split}")
    print(f"Num records: {len(records)}")
    print(f"Unique videos: {len(video_counts)}")
    print(f"Avg questions/video: {len(records) / max(1, len(video_counts)):.2f}")
    print(f"Avg transcript tokens: {statistics.mean(transcript_token_lengths):.2f}")
    print(f"Median transcript tokens: {statistics.median(transcript_token_lengths):.2f}")
    print(f"Avg num choices: {statistics.mean(num_choices):.2f}")
    print(f"% with start/end times: {100.0 * sum(has_time) / len(has_time):.2f}")
    print("Subjects:")
    for subject, count in subject_counts.most_common():
        print(f"  - {subject}: {count}")

    print("\nSample record keys:")
    print(sorted(records[0].keys()))
    print("\nSample canonical record:")
    sample = records[0].copy()
    sample["raw"] = "<omitted>"
    print(sample)


if __name__ == "__main__":
    main()
