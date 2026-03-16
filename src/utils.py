from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


def normalize_text(text: str) -> str:
    text = text or ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def simple_tokenize(text: str) -> List[str]:
    text = normalize_text(text)
    return re.findall(r"[a-z0-9_\-\.]+", text)


def sliding_char_windows(text: str, window_char_len: int, window_char_stride: int) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= window_char_len:
        return [text]

    windows = []
    start = 0
    while start < len(text):
        end = min(len(text), start + window_char_len)
        chunk = text[start:end].strip()
        if chunk:
            windows.append(chunk)
        if end == len(text):
            break
        start += window_char_stride
    return windows


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass
class Timer:
    start_time: float = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    @property
    def elapsed_s(self) -> float:
        return time.perf_counter() - self.start_time
