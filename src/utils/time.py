from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class Timer:
    start_time: float = 0.0

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    @property
    def elapsed_s(self) -> float:
        return time.perf_counter() - self.start_time
