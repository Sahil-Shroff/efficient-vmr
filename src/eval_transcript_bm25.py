from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

from rank_bm25 import BM25Okapi
from tqdm import tqdm

from .load_videommlu import load_records
from .utils import Timer, ensure_dir, normalize_text, simple_tokenize, sliding_char_windows


def build_windows(transcript: str, window_char_len: int, window_char_stride: int) -> List[str]:
    return sliding_char_windows(
        transcript,
        window_char_len=window_char_len,
        window_char_stride=window_char_stride,
    )


def retrieve_top_k(windows: List[str], query: str, top_k: int) -> List[Tuple[int, str, float]]:
    if not windows:
        return []
    tokenized_windows = [simple_tokenize(w) for w in windows]
    bm25 = BM25Okapi(tokenized_windows)
    scores = bm25.get_scores(simple_tokenize(query))
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [(idx, windows[idx], float(score)) for idx, score in ranked]


def lexical_choice_score(choice: str, evidence_text: str) -> float:
    choice_tokens = set(simple_tokenize(choice))
    evidence_tokens = set(simple_tokenize(evidence_text))
    if not choice_tokens:
        return 0.0
    overlap = len(choice_tokens & evidence_tokens)
    return overlap / max(1, len(choice_tokens))


def split_sentences(text: str) -> List[str]:
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
    norm_text = normalize_text(text)
    return bool(norm_answer) and norm_answer in norm_text


def extract_answer(question: str, retrieved_windows: List[Tuple[int, str, float]]) -> Tuple[str, float]:
    if not retrieved_windows:
        return "", 0.0

    top_window = retrieved_windows[0][1]
    sentences = split_sentences(top_window) or [top_window]
    scored = [(sentence, token_overlap_score(question, sentence)) for sentence in sentences]
    best_sentence, best_score = max(scored, key=lambda item: item[1])
    return best_sentence, best_score


def predict_answer(question: str, choices: List[str], retrieved_windows: List[Tuple[int, str, float]]) -> Tuple[int, List[float], float]:
    evidence = "\n".join([w for _, w, _ in retrieved_windows])
    choice_scores = []
    for choice in choices:
        score = lexical_choice_score(choice, evidence)
        choice_scores.append(score)

    if not choice_scores:
        return -1, [], 0.0

    pred_idx = max(range(len(choice_scores)), key=lambda i: choice_scores[i])
    sorted_scores = sorted(choice_scores, reverse=True)
    margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0]
    return pred_idx, choice_scores, margin


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train")
    parser.add_argument("--window-char-len", type=int, default=700)
    parser.add_argument("--window-char-stride", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--max-examples", type=int, default=0)
    args = parser.parse_args()

    out_dir = ensure_dir(args.output_dir)
    records = load_records(split=args.split)
    if args.max_examples > 0:
        records = records[: args.max_examples]

    if not records:
        raise ValueError("No records loaded.")

    preds_path = out_dir / f"predictions_{args.split}.jsonl"
    summary_path = out_dir / f"summary_{args.split}.json"

    correct = 0
    total = 0
    latencies_ms = []
    retrieval_latencies_ms = []
    llm_latencies_ms = []
    margins = []
    retrieval_hits = 0
    exact_match_correct = 0
    contains_correct = 0

    with preds_path.open("w", encoding="utf-8") as wf:
        for row in tqdm(records, desc="Evaluating"):
            transcript = row.get("transcript", "") or ""
            question = row.get("question", "") or ""
            choices = row.get("choices", []) or []
            answer = row.get("answer")

            query = question + " " + " ".join(str(c) for c in choices)
            windows = build_windows(transcript, args.window_char_len, args.window_char_stride)

            with Timer() as total_timer:
                with Timer() as retrieval_timer:
                    retrieved = retrieve_top_k(windows, query, args.top_k)
                retrieval_latency_ms = retrieval_timer.elapsed_s * 1000.0

                # Placeholder for a future LLM call. The current baseline is retrieval-only.
                with Timer() as llm_timer:
                    pred_idx, choice_scores, margin = predict_answer(question, choices, retrieved)
                    pred_text, pred_text_score = extract_answer(question, retrieved)
                llm_latency_ms = llm_timer.elapsed_s * 1000.0
            latency_ms = total_timer.elapsed_s * 1000.0

            gold_answer_text = str(answer) if answer is not None else ""
            retrieval_hit = int(answer_in_text(gold_answer_text, "\n".join(text for _, text, _ in retrieved)))
            retrieval_hits += retrieval_hit

            is_correct = int(pred_idx == answer) if answer is not None and pred_idx >= 0 else 0
            correct += is_correct
            exact_match = int(normalize_text(pred_text) == normalize_text(gold_answer_text)) if gold_answer_text else 0
            contains_match = int(answer_in_text(gold_answer_text, pred_text))
            exact_match_correct += exact_match
            contains_correct += contains_match
            total += 1
            latencies_ms.append(latency_ms)
            retrieval_latencies_ms.append(retrieval_latency_ms)
            llm_latencies_ms.append(llm_latency_ms)
            margins.append(margin)

            rec = {
                "video_id": row.get("video_id"),
                "question_id": row.get("question_id"),
                "subject": row.get("subject"),
                "question": question,
                "choices": choices,
                "gold_answer": answer,
                "pred_answer": pred_idx,
                "is_correct": is_correct,
                "gold_answer_text": gold_answer_text,
                "pred_answer_text": pred_text,
                "pred_answer_text_score": pred_text_score,
                "retrieval_hit_at_k": retrieval_hit,
                "answer_exact_match": exact_match,
                "answer_contains_match": contains_match,
                "choice_scores": choice_scores,
                "confidence_margin": margin,
                "retrieval_time_ms": retrieval_latency_ms,
                "llm_time_ms": llm_latency_ms,
                "total_latency_ms": latency_ms,
                "retrieved": [
                    {"rank": rank + 1, "window_idx": idx, "score": score, "text": text}
                    for rank, (idx, text, score) in enumerate(retrieved)
                ],
            }
            wf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    lat_sorted = sorted(latencies_ms)
    summary = {
        "split": args.split,
        "num_examples": total,
        "accuracy": (correct / total) if total else 0.0,
        "answer_exact_match_accuracy": (exact_match_correct / total) if total else 0.0,
        "answer_contains_accuracy": (contains_correct / total) if total else 0.0,
        "retrieval_recall_at_k": (retrieval_hits / total) if total else 0.0,
        "avg_retrieval_time_ms": sum(retrieval_latencies_ms) / len(retrieval_latencies_ms) if retrieval_latencies_ms else 0.0,
        "avg_llm_time_ms": sum(llm_latencies_ms) / len(llm_latencies_ms) if llm_latencies_ms else 0.0,
        "avg_latency_ms": sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0,
        "p50_latency_ms": lat_sorted[len(lat_sorted) // 2] if lat_sorted else 0.0,
        "p95_latency_ms": lat_sorted[min(len(lat_sorted) - 1, math.floor(0.95 * len(lat_sorted)))] if lat_sorted else 0.0,
        "avg_confidence_margin": sum(margins) / len(margins) if margins else 0.0,
        "window_char_len": args.window_char_len,
        "window_char_stride": args.window_char_stride,
        "top_k": args.top_k,
        "max_examples": args.max_examples,
        "predictions_file": str(preds_path),
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
