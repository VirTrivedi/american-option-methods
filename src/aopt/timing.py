"""Wall-clock timing helper for error-vs-runtime comparisons."""
from __future__ import annotations
import time
from collections.abc import Callable
from typing import TypeVar


T = TypeVar("T")


def median_runtime(fn: Callable[[], T], repeats: int = 5, warmup: int = 1) -> tuple[T, float]:
    """Run ``fn`` ``warmup + repeats`` times; return its last result and the median seconds of the timed runs."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    for _ in range(warmup):
        fn()
    times: list[float] = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    times.sort()
    mid = len(times) // 2
    median = times[mid] if len(times) % 2 else 0.5 * (times[mid - 1] + times[mid])
    return result, median  # type: ignore[return-value]