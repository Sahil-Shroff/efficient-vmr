from __future__ import annotations

import html
import re


TAG_RE = re.compile(r"<[^>]+>")


def normalize_text(text: str) -> str:
    text = text or ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def simple_tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_\-\.]+", normalize_text(text))


def strip_markup(text: str) -> str:
    text = html.unescape(text or "")
    text = TAG_RE.sub("", text)
    return normalize_whitespace(text)
