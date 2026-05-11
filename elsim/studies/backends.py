"""
Execution helpers for repeating independent Monte Carlo batches.

Serial execution is always available for :func:`elsim.studies.runner.run_batched`.
"""

from __future__ import annotations

from typing import Callable, Sequence, TypeVar

T = TypeVar("T")


class SerialBackend:
    """Run ``fn()`` ``n`` times in the current process."""

    def map_repeat(self, fn: Callable[[], T], n: int) -> list[T]:
        if n < 0:
            raise ValueError("n must be non-negative")
        return [fn() for _ in range(n)]

    def map_each(self, fns: Sequence[Callable[[], T]]) -> list[T]:
        """Invoke each zero-argument callable once, in order, and collect results."""
        return [fn() for fn in fns]
