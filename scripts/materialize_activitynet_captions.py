import argparse, json, os, random
from pathlib import Path

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True, help="Directory containing train.json, val_1.json, etc.")
    ap.add_argument("--out_dir", required=True, help="Output dir for dataset artifacts")
    ap.add_argument("--min_duration_sec", type=float, default=600.0, help="Filter for long videos (default 10 min)")
    ap.add_argument("--max_videos", type=int, default=200, help="Cap number of videos")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    random.seed(args.seed)
    raw = Path(args.raw_dir)
    out = Path(args.out_dir)

    # Prefer train + val_1
    sources = []
    for name in ["train.json", "val_1.json"]:
        p = raw / name
        if p.exists():
            sources.append((name, load_json(p)))
    if not sources:
        raise FileNotFoundError("Could not find train.json or val_1.json in raw_dir")

    # Merge dicts; if duplicates, later sources overwrite (fine)
    merged = {}
    for name, d in sources:
        merged.update(d)

    # Filter long videos
    items = []
    for vid, obj in merged.items():
        dur = float(obj.get("duration", 0.0))
        if dur >= args.min_duration_sec:
            items.append((vid, obj))
    items.sort(key=lambda x: float(x[1].get("duration", 0.0)), reverse=True)

    # Cap videos (take longest first, then sample if too many)
    if len(items) > args.max_videos:
        items = items[:args.max_videos]

    metadata = []
    queries = []
    qrels = []

    transcripts_clean_dir = out / "transcripts_clean"
    transcripts_clean_dir.mkdir(parents=True, exist_ok=True)

    for vid, obj in items:
        dur = float(obj["duration"])
        sentences = obj.get("sentences", [])
        timestamps = obj.get("timestamps", [])

        # Build transcripts_clean/<vid>.jsonl from captions
        seg_rows = []
        for i, (sent, ts) in enumerate(zip(sentences, timestamps)):
            start, end = float(ts[0]), float(ts[1])
            text = (sent or "").strip()
            if not text:
                continue
            seg_rows.append({"start": start, "end": end, "text": text})

            # Use each caption as a query (qid stable)
            qid = f"{vid}_{i:04d}"
            queries.append({"qid": qid, "video_id": vid, "query": text, "type": "activitynet_caption"})
            qrels.append({"qid": qid, "video_id": vid, "start": start, "end": end})

        write_jsonl(transcripts_clean_dir / f"{vid}.jsonl", seg_rows)

        metadata.append({
            "video_id": vid,
            "duration": dur,
            "source": "activitynet_captions",
            "num_captions": len(seg_rows)
        })

    # Write dataset-level files
    write_jsonl(out / "metadata.jsonl", metadata)
    write_jsonl(out / "queries.jsonl", queries)
    write_jsonl(out / "qrels.jsonl", qrels)

    # Also write a convenient list file
    (out / "raw").mkdir(parents=True, exist_ok=True)
    with open(out / "raw" / "video_list.json", "w") as f:
        json.dump([m["video_id"] for m in metadata], f, indent=2)

    print(f"Done. Videos: {len(metadata)} | Queries: {len(queries)}")
    print(f"Output at: {out}")

if __name__ == "__main__":
    main()
