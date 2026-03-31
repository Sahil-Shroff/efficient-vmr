#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import load_tvr_queries
from src.utils import ensure_dir
from src.visual import ClipFeatureEncoder, sample_video_frames, save_visual_feature_file


def _discover_video_paths(video_dir: str | Path, suffixes: set[str]) -> dict[str, Path]:
    video_paths: dict[str, Path] = {}
    for path in sorted(Path(video_dir).rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        video_paths.setdefault(path.stem, path)
    return video_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CLIP frame features for local TVR video clips.")
    parser.add_argument("--data-dir", default="data/tvr")
    parser.add_argument("--split", default="")
    parser.add_argument("--query-path", default="")
    parser.add_argument("--video-dir", default="data/tvr/videos")
    parser.add_argument("--output-dir", default="data/tvr/processed/visual_features")
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default="")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sample-stride-sec", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--video-suffixes", default=".mp4,.mkv,.avi,.mov,.webm")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    suffixes = {token.strip().lower() for token in args.video_suffixes.split(",") if token.strip()}
    output_dir = ensure_dir(args.output_dir)
    video_paths = _discover_video_paths(args.video_dir, suffixes)
    if not video_paths:
        raise FileNotFoundError(f"No video files found under {args.video_dir}")

    if args.split or args.query_path:
        split = args.split or "custom"
        queries = load_tvr_queries(split=split, data_dir=args.data_dir, query_path=args.query_path)
        vid_names = sorted({str(query["vid_name"]) for query in queries})
    else:
        vid_names = sorted(video_paths)

    if args.max_videos > 0:
        vid_names = vid_names[: args.max_videos]

    encoder = ClipFeatureEncoder(
        model_name=args.model_name,
        device=args.device or None,
        batch_size=args.batch_size,
    )
    num_built = 0
    for vid_name in tqdm(vid_names, desc="TVR visual features"):
        video_path = video_paths.get(vid_name)
        if video_path is None:
            continue

        output_path = output_dir / f"{vid_name}.npz"
        if output_path.exists() and not args.overwrite:
            continue

        timestamps, frames = sample_video_frames(
            video_path,
            sample_stride_sec=args.sample_stride_sec,
            max_frames=args.max_frames,
        )
        if len(frames) == 0:
            continue

        embeddings = encoder.encode_images(frames)
        save_visual_feature_file(output_path, timestamps=timestamps, embeddings=embeddings)
        num_built += 1

    print(f"Built {num_built} visual feature files under {output_dir}")


if __name__ == "__main__":
    main()
