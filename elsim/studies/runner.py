"""
Batched Monte Carlo execution and simple result merging.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, Iterable, TypeVar

from .backends import SerialBackend

T = TypeVar("T")


def run_batched(
    batch_fn: Callable[[int], T],
    n_trials: int,
    batch_size: int,
    *,
    backend=None,
) -> list[T]:
    """
    Run a trial batch worker an integer number of times.

    ``batch_fn(k)`` is invoked with ``k == batch_size`` for each full batch,
    and once more with ``k == n_trials % batch_size`` when the remainder is
    non-zero.

    Parameters
    ----------
    batch_fn : callable
        ``batch_fn(batch_size) -> partial result`` for one batch.
    n_trials : int
        Total number of trials across all batches.
    batch_size : int
        Preferred batch size (must be positive).
    backend : object with ``map_repeat(fn, n) -> list``, optional
        Defaults to :class:`elsim.studies.backends.SerialBackend` inside
        :func:`map_repeat` for the full-sized batches only; the remainder batch
        always runs in-process.

    Returns
    -------
    list
        One return value per batch invocation (length ``ceil(n_trials / batch_size)``).
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if n_trials < 0:
        raise ValueError("n_trials must be non-negative")
    if n_trials == 0:
        return []

    n_full, rem = divmod(n_trials, batch_size)
    if backend is None:
        backend = SerialBackend()

    parts: list[T] = backend.map_repeat(lambda: batch_fn(batch_size), n_full)
    if rem:
        parts.append(batch_fn(rem))
    return parts


def merge_counters(partials: Iterable[Counter]) -> Counter:
    """Sum a sequence of :class:`~collections.Counter` objects."""
    total: Counter = Counter()
    for c in partials:
        total.update(c)
    return total
