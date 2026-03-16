from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize subtitle download manifest coverage and failures.")
    parser.add_argument("--manifest-path", default="data/subtitles/subtitle_manifest.jsonl")
    parser.add_argument("--limit-failures", type=int, default=20)
    args = parser.parse_args()

    manifest_path = Path(args.manifest_path)
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    counts = Counter(row.get("status", "unknown") for row in rows)

    success_count = counts["success_human"] + counts["success_auto"]
    summary = {
        "manifest_path": str(manifest_path),
        "num_rows": len(rows),
        "success_human": counts["success_human"],
        "success_auto": counts["success_auto"],
        "unavailable": counts["unavailable"],
        "failed": counts["failed"],
        "success_percentage": (100.0 * success_count / len(rows)) if rows else 0.0,
    }
    print(json.dumps(summary, indent=2))

    failures = [row for row in rows if row.get("status") in {"failed", "unavailable"}]
    if failures:
        print("\nSample failures:")
        for row in failures[: args.limit_failures]:
            print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
