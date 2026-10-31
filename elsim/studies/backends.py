"""
Execution backends for repeating independent Monte Carlo batches.

Serial execution is always available.  Parallel execution uses Joblib when
installed (same optional dependency as the example scripts).
"""

from __future__ import annotations

from typing import Any, Callable, Sequence, TypeVar

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


class JoblibBackend:
    """
    Run independent calls with :class:`joblib.Parallel`.

    Parameters mirror the common ``Parallel`` constructor; see Joblib docs for
    ``prefer``, ``backend``, etc.

    Raises
    ------
    ImportError
        If ``joblib`` is not installed.
    """

    def __init__(self, n_jobs: int = -1, verbose: int = 0, **parallel_kwargs: Any):
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.parallel_kwargs = parallel_kwargs

    def map_repeat(self, fn: Callable[[], T], n: int) -> list[T]:
        if n < 0:
            raise ValueError("n must be non-negative")
        try:
            from joblib import Parallel, delayed
        except ImportError as exc:
            raise ImportError(
                "JoblibBackend requires the 'joblib' package "
                "(install with pip install 'elsim[examples]' or pip install joblib)."
            ) from exc

        jobs = [delayed(fn)() for _ in range(n)]
        return Parallel(
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            **self.parallel_kwargs,
        )(jobs)

    def map_each(self, fns: Sequence[Callable[[], T]]) -> list[T]:
        """
        Invoke each zero-argument callable once (typical pattern: a list of
        :func:`functools.partial` objects), preserving order of ``fns``.
        """
        try:
            from joblib import Parallel, delayed
        except ImportError as exc:
            raise ImportError(
                "JoblibBackend requires the 'joblib' package "
                "(install with pip install 'elsim[examples]' or pip install joblib)."
            ) from exc

        jobs = [delayed(fn)() for fn in fns]
        return Parallel(
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            **self.parallel_kwargs,
        )(jobs)
