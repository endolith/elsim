"""
Condorcet-efficiency tallies used by several Merrill (1984) example scripts.

These are thin wrappers around existing ``methods`` and ``strategies`` helpers
so driver loops can ``counter.update(...)`` in one line.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, Mapping, Optional

import numpy as np

from elsim.methods import approval, condorcet
from elsim.strategies import approval_optimal

RankedMethod = Callable[..., Optional[int]]
RatedMethod = Callable[..., Optional[int]]


def approval_at_optimal(utilities: np.ndarray, tiebreaker: str = "random") -> Optional[int]:  # noqa: UP045
    """
    Rated-method helper: build an optimal approval ballot, then apply :func:`elsim.methods.approval`.

    Intended for use in a ``rated_methods`` mapping passed to
    :func:`tally_condorcet_agreement` or the spatial sweep helpers, so scripts
    stay declarative without repeating lambdas.
    """
    return approval(approval_optimal(utilities), tiebreaker)


def tally_condorcet_agreement(
    rankings: np.ndarray,
    utilities: np.ndarray,
    ranked_methods: Mapping[str, RankedMethod],
    rated_methods: Mapping[str, RatedMethod],
    *,
    tiebreaker: str = "random",
) -> Counter:
    """
    Count whether each method agrees with the Condorcet winner for one election.

    If there is no Condorcet winner, returns an empty counter.

    Parameters
    ----------
    rankings : array_like
        Honest (or strategic) rankings, shape ``(n_voters, n_cands)``.
    utilities : array_like
        Utilities aligned with ``rankings``, shape ``(n_voters, n_cands)``.
    ranked_methods, rated_methods
        Name to callable maps; ranked methods take ``(rankings, tiebreaker=...)``;
        rated methods take ``(utilities, tiebreaker=...)``.
    tiebreaker : str, optional
        Passed through to each method callable.

    Returns
    -------
    collections.Counter
        Includes key ``\"CW\"`` when a Condorcet winner exists, plus one key per
        supplied method name when that method's winner matches the Condorcet
        winner.
    """
    cw = condorcet(rankings)
    if cw is None:
        return Counter()

    out: Counter = Counter()
    out["CW"] += 1

    for name, fn in ranked_methods.items():
        if fn(rankings, tiebreaker=tiebreaker) == cw:
            out[name] += 1

    for name, fn in rated_methods.items():
        if fn(utilities, tiebreaker=tiebreaker) == cw:
            out[name] += 1

    return out
