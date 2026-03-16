from __future__ import annotations

import argparse
import json
import math
import re

from rank_bm25 import BM25Okapi
from tqdm import tqdm

from ..data import load_flattened_examples
from ..utils import Timer, ensure_dir, normalize_text, simple_tokenize


def build_windows(transcript: str, window_char_len: int, window_char_stride: int) -> list[str]:
    transcript = (transcript or "").strip()
    if not transcript:
        return []
    if len(transcript) <= window_char_len:
        return [transcript]

    windows: list[str] = []
    start = 0
    while start < len(transcript):
        end = min(len(transcript), start + window_char_len)
        chunk = transcript[start:end].strip()
        if chunk:
            windows.append(chunk)
        if end == len(transcript):
            break
        start += window_char_stride
    return windows


def retrieve_top_k(windows: list[str], query: str, top_k: int) -> list[tuple[int, str, float]]:
    if not windows:
        return []
    bm25 = BM25Okapi([simple_tokenize(window) for window in windows])
    scores = bm25.get_scores(simple_tokenize(query))
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]
    return [(idx, windows[idx], float(score)) for idx, score in ranked]


def split_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]


def token_overlap_score(query: str, text: str) -> float:
    query_tokens = set(simple_tokenize(query))
    text_tokens = set(simple_tokenize(text))
    if not query_tokens or not text_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / max(1, len(query_tokens))


def answer_in_text(answer: str, text: str) -> bool:
    norm_answer = normalize_text(answer)
    return bool(norm_answer) and norm_answer in normalize_text(text)


def extract_answer(question: str, retrieved_windows: list[tuple[int, str, float]]) -> tuple[str, float]:
    if not retrieved_windows:
        return "", 0.0
    sentences = split_sentences(retrieved_windows[0][1]) or [retrieved_windows[0][1]]
    scored = [(sentence, token_overlap_score(question, sentence)) for sentence in sentences]
    return max(scored, key=lambda item: item[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Legacy QA-oriented transcript retrieval baseline.")
    parser.add_argument("--split", default="Video_MMLU")
    parser.add_argument("--window-char-len", type=int, default=700)
    parser.add_argument("--window-char-stride", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output-dir", default="outputs/legacy_qa")
    parser.add_argument("--max-examples", type=int, default=0)
    args = parser.parse_args()

    rows = load_flattened_examples(split=args.split)
    if args.max_examples > 0:
        rows = rows[: args.max_examples]
    if not rows:
        raise ValueError("No flattened rows available.")

    out_dir = ensure_dir(args.output_dir)
    predictions_path = out_dir / f"predictions_{args.split}.jsonl"
    summary_path = out_dir / f"summary_{args.split}.json"

    retrieval_hits = 0
    answer_contains_hits = 0
    latencies: list[float] = []

    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in tqdm(rows, desc="Legacy QA baseline"):
            transcript = str(row.get("transcript", ""))
            question = str(row.get("question", ""))
            answer = str(row.get("answer", "") or "")
            windows = build_windows(
                transcript,
                window_char_len=args.window_char_len,
                window_char_stride=args.window_char_stride,
            )
            with Timer() as timer:
                retrieved = retrieve_top_k(windows, question, args.top_k)
                predicted_text, predicted_score = extract_answer(question, retrieved)
            latency_ms = timer.elapsed_s * 1000.0
            latencies.append(latency_ms)

            retrieval_hit = int(answer_in_text(answer, "\n".join(text for _, text, _ in retrieved)))
            contains_hit = int(answer_in_text(answer, predicted_text))
            retrieval_hits += retrieval_hit
            answer_contains_hits += contains_hit

            handle.write(
                json.dumps(
                    {
                        "video_id": row.get("video_id"),
                        "question_id": row.get("question_id"),
                        "qa_type": row.get("qa_type"),
                        "question": question,
                        "gold_answer": answer,
                        "predicted_answer_text": predicted_text,
                        "predicted_answer_text_score": predicted_score,
                        "retrieval_hit_at_k": retrieval_hit,
                        "answer_contains_match": contains_hit,
                        "latency_ms": latency_ms,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    latencies_sorted = sorted(latencies)
    summary = {
        "split": args.split,
        "num_examples": len(rows),
        "retrieval_recall_at_k": retrieval_hits / len(rows),
        "answer_contains_accuracy": answer_contains_hits / len(rows),
        "avg_latency_ms": sum(latencies) / len(latencies),
        "p50_latency_ms": latencies_sorted[len(latencies_sorted) // 2],
        "p95_latency_ms": latencies_sorted[min(len(latencies_sorted) - 1, math.floor(0.95 * len(latencies_sorted)))],
        "predictions_file": str(predictions_path),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
