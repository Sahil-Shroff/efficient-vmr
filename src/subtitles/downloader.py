from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm
from yt_dlp import DownloadError, YoutubeDL

from ..data import extract_unique_video_ids, load_raw_video_rows
from ..utils import ensure_dir, read_jsonl


DEFAULT_MANIFEST_PATH = Path("data/subtitles/subtitle_manifest.jsonl")
DEFAULT_RAW_DIR = Path("data/subtitles/raw")
PREFERRED_ENGLISH_LANGS = ["en", "en-US", "en-GB", "en-CA", "en-AU", "en-orig"]


def _load_existing_manifest(path: Path) -> dict[str, dict[str, Any]]:
    manifest_by_video: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        video_id = row.get("video_id")
        if video_id:
            manifest_by_video[video_id] = row
    return manifest_by_video


def _write_manifest(path: Path, manifest_by_video: dict[str, dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for video_id in sorted(manifest_by_video):
            handle.write(json.dumps(manifest_by_video[video_id], ensure_ascii=False) + "\n")


def _download_subtitles_once(
    *,
    video_id: str,
    raw_dir: Path,
    retries: int,
    request_sleep_seconds: float,
) -> dict[str, Any]:
    url = f"https://www.youtube.com/watch?v={video_id}"
    temp_pattern = str(raw_dir / f"{video_id}.%(ext)s")
    for stale_file in raw_dir.glob(f"{video_id}*.vtt"):
        stale_file.unlink()
    options = {
        "quiet": True,
        "noprogress": True,
        "noplaylist": True,
        "skip_download": True,
        "retries": retries,
        "socket_timeout": 20,
        "sleep_interval_requests": request_sleep_seconds,
        "subtitleslangs": PREFERRED_ENGLISH_LANGS,
        "subtitlesformat": "vtt",
        "outtmpl": temp_pattern,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "js_runtimes": {"node": {}},
        "remote_components": ["ejs:github"],
    }
    with YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)

    requested_subtitles = info.get("requested_subtitles") or {}
    ranked_subtitles: list[tuple[int, bool, str, Path]] = []
    for language, subtitle_info in requested_subtitles.items():
        filepath = subtitle_info.get("filepath")
        if not filepath:
            continue
        candidate_path = Path(filepath)
        if not candidate_path.exists():
            continue
        is_auto = "kind=asr" in str(subtitle_info.get("url", ""))
        try:
            language_rank = PREFERRED_ENGLISH_LANGS.index(language)
        except ValueError:
            language_rank = len(PREFERRED_ENGLISH_LANGS)
        ranked_subtitles.append((0 if not is_auto else 1, language_rank, language, candidate_path))

    if not ranked_subtitles:
        return {
            "video_id": video_id,
            "status": "unavailable",
            "subtitle_path": None,
            "subtitle_type": None,
            "error_message": None,
        }

    ranked_subtitles.sort()
    is_auto = ranked_subtitles[0][0] == 1
    selected_path = ranked_subtitles[0][3]
    subtitle_kind = "auto" if is_auto else "human"
    target_path = raw_dir / f"{video_id}.{subtitle_kind}.en.vtt"
    if selected_path != target_path:
        if target_path.exists():
            target_path.unlink()
        selected_path.rename(target_path)
    for _, _, _, extra_path in ranked_subtitles[1:]:
        if extra_path.exists():
            extra_path.unlink()

    return {
        "video_id": video_id,
        "status": "success_auto" if is_auto else "success_human",
        "subtitle_path": str(target_path),
        "subtitle_type": subtitle_kind,
        "error_message": None,
    }


def _download_single_video(
    *,
    video_id: str,
    raw_dir: Path,
    retries: int,
    request_sleep_seconds: float,
) -> dict[str, Any]:
    try:
        return _download_subtitles_once(
            video_id=video_id,
            raw_dir=raw_dir,
            retries=retries,
            request_sleep_seconds=request_sleep_seconds,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "video_id": video_id,
            "status": "failed",
            "subtitle_path": None,
            "subtitle_type": None,
            "error_message": str(exc),
        }


def _is_rate_limited(result: dict[str, Any]) -> bool:
    error_message = str(result.get("error_message") or "")
    return "HTTP Error 429" in error_message


def _resolve_existing_download(video_id: str, raw_dir: Path) -> dict[str, Any] | None:
    for subtitle_kind, status in (("human", "success_human"), ("auto", "success_auto")):
        candidate = raw_dir / f"{video_id}.{subtitle_kind}.en.vtt"
        if candidate.exists():
            return {
                "video_id": video_id,
                "status": status,
                "subtitle_path": str(candidate),
                "subtitle_type": subtitle_kind,
                "error_message": None,
            }
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Download English YouTube subtitles for Video-MMLU videos.")
    parser.add_argument("--split", default="Video_MMLU")
    parser.add_argument("--video-ids-path", default="")
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--request-sleep-seconds", type=float, default=0.5)
    parser.add_argument("--rate-limit-retries", type=int, default=2)
    parser.add_argument("--rate-limit-sleep-seconds", type=float, default=10.0)
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()

    if args.video_ids_path:
        video_ids = [
            line.strip()
            for line in Path(args.video_ids_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        raw_rows = load_raw_video_rows(split=args.split)
        video_ids = extract_unique_video_ids(raw_rows)
    if args.max_videos > 0:
        video_ids = video_ids[: args.max_videos]
    raw_dir = ensure_dir(args.raw_dir)
    manifest_path = Path(args.manifest_path)
    manifest_by_video = _load_existing_manifest(manifest_path)

    counts = {"success_human": 0, "success_auto": 0, "unavailable": 0, "failed": 0}

    for video_id in tqdm(video_ids, desc="Downloading subtitles"):
        existing = manifest_by_video.get(video_id)
        existing_file = _resolve_existing_download(video_id, raw_dir)

        should_skip = False
        if not args.force:
            if existing_file:
                manifest_by_video[video_id] = existing_file
                should_skip = True
            elif existing and existing.get("status") in {"success_human", "success_auto"}:
                should_skip = True
            elif (
                existing
                and existing.get("status") in {"failed", "unavailable"}
                and not args.retry_failed
            ):
                should_skip = True

        if should_skip:
            status = manifest_by_video.get(video_id, existing or {})
            if status.get("status") in counts:
                counts[status["status"]] += 1
            continue

        result = _download_single_video(
            video_id=video_id,
            raw_dir=raw_dir,
            retries=args.retries,
            request_sleep_seconds=args.request_sleep_seconds,
        )
        retry_count = 0
        while _is_rate_limited(result) and retry_count < args.rate_limit_retries:
            retry_count += 1
            time.sleep(args.rate_limit_sleep_seconds * retry_count)
            result = _download_single_video(
                video_id=video_id,
                raw_dir=raw_dir,
                retries=args.retries,
                request_sleep_seconds=args.request_sleep_seconds,
            )
        manifest_by_video[video_id] = result
        if result["status"] in counts:
            counts[result["status"]] += 1
        _write_manifest(manifest_path, manifest_by_video)
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    _write_manifest(manifest_path, manifest_by_video)

    total = len(video_ids)
    success_total = counts["success_human"] + counts["success_auto"]
    pct = 100.0 * success_total / total if total else 0.0
    print(
        json.dumps(
            {
                "num_videos": total,
                "success_human": counts["success_human"],
                "success_auto": counts["success_auto"],
                "unavailable": counts["unavailable"],
                "failed": counts["failed"],
                "success_percentage": pct,
                "manifest_path": str(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
