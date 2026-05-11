"""
Scalar social-utility totals for Monte Carlo scripts (Merrill, Weber, etc.).

These return per-election increments as plain floats so callers can accumulate
into :class:`collections.Counter` objects keyed by scenario (as in Merrill
Table 4) or by ``(method, n_cands)`` via nested counters.
"""

from __future__ import annotations

import random
from typing import Callable, Mapping, Optional

import numpy as np

from elsim.methods import utility_winner

RankedMethod = Callable[..., Optional[int]]
RatedMethod = Callable[..., Optional[int]]


def spatial_random_reference_utility_updates(
    utilities: np.ndarray,
    rankings: np.ndarray,
    ranked_methods: Mapping[str, RankedMethod],
    rated_methods: Mapping[str, RatedMethod],
    *,
    tiebreaker: str = "random",
) -> dict[str, float]:
    """
    Total utility (summed over voters) for each method winner plus random baseline.

    Matches the Merrill (1984) spatial social-utility-efficiency figures: pick
    ``RW`` with ``random.randint``, accumulate rated and ranked method winners,
    and use the same per-winner column sum as ``utilities.sum(axis=0)[w]``.
    """
    n_cands = utilities.shape[1]
    rw = random.randint(0, n_cands - 1)
    out: dict[str, float] = {"RW": float(utilities.sum(axis=0)[rw])}

    for name, fn in rated_methods.items():
        w = fn(utilities, tiebreaker=tiebreaker)
        out[name] = float(utilities.sum(axis=0)[w])

    for name, fn in ranked_methods.items():
        w = fn(rankings, tiebreaker=tiebreaker)
        out[name] = float(utilities.sum(axis=0)[w])

    return out


def random_society_utility_updates(
    utilities: np.ndarray,
    rankings: np.ndarray,
    ranked_methods: Mapping[str, RankedMethod],
    rated_methods: Mapping[str, RatedMethod],
    *,
    tiebreaker: str = "random",
    uw_key: str = "UW",
    utility_winner_tiebreaker: Optional[str] = "random",  # noqa: UP045
) -> dict[str, float]:
    """
    Utility totals for Merrill-style random societies (Table 3 / Fig 3).

    Parameters
    ----------
    utility_winner_tiebreaker
        If ``None``, call ``utility_winner(utilities)`` with no tiebreaker
        (Weber-style scripts). Otherwise pass through to ``utility_winner``.
    """
    if utility_winner_tiebreaker is None:
        uw = utility_winner(utilities)
    else:
        uw = utility_winner(utilities, tiebreaker=utility_winner_tiebreaker)
    out: dict[str, float] = {uw_key: float(utilities.sum(axis=0)[uw])}

    for name, fn in rated_methods.items():
        w = fn(utilities, tiebreaker=tiebreaker)
        out[name] = float(utilities.sum(axis=0)[w])

    for name, fn in ranked_methods.items():
        w = fn(rankings, tiebreaker=tiebreaker)
        out[name] = float(utilities.sum(axis=0)[w])

    return out


def ranked_rated_utility_updates(
    utilities: np.ndarray,
    rankings: np.ndarray,
    ranked_methods: Mapping[str, RankedMethod],
    rated_methods: Mapping[str, RatedMethod],
    *,
    tiebreaker: str = "random",
) -> dict[str, float]:
    """Per-election utility totals for ranked and rated methods only (no UW/RW)."""
    out: dict[str, float] = {}
    for name, fn in rated_methods.items():
        w = fn(utilities, tiebreaker=tiebreaker)
        out[name] = float(utilities.sum(axis=0)[w])
    for name, fn in ranked_methods.items():
        w = fn(rankings, tiebreaker=tiebreaker)
        out[name] = float(utilities.sum(axis=0)[w])
    return out
