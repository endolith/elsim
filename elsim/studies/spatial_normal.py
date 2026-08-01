"""
Spatial normal-electorate Monte Carlo sweeps (Merrill-style figures).

These functions implement the common ``for each election: for each n_cands:``
pattern so example scripts only declare parameters and method maps.
"""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

from elsim.elections import normal_electorate, normed_dist_utilities
from elsim.strategies import honest_rankings

from .condorcet_metrics import RankedMethod, RatedMethod, tally_condorcet_agreement
from .social_utility import spatial_random_reference_utility_updates


def accumulate_spatial_condorcet_by_ncands(
    n_elections: int,
    *,
    n_voters: int,
    n_cands_list: Sequence[int],
    dims: int,
    corr: float,
    disp: float,
    ranked_methods: Mapping[str, RankedMethod],
    rated_methods: Mapping[str, RatedMethod],
    tiebreaker: str = "random",
) -> dict[str, Counter]:
    """
    Run elections on a spatial normal model and tally Condorcet agreement by ``n_cands``.

    For each of ``n_elections`` iterations, draws one electorate per entry in
    ``n_cands_list`` (same pattern as Merrill figures 2.c / 2.d).
    """
    keys = ranked_methods.keys() | rated_methods.keys() | {"CW"}
    out: dict[str, Counter] = {k: Counter() for k in keys}

    for _ in range(n_elections):
        for n_cands in n_cands_list:
            v, c = normal_electorate(n_voters, n_cands, dims=dims, corr=corr, disp=disp)
            utilities = normed_dist_utilities(v, c)
            rankings = honest_rankings(utilities)
            delta = tally_condorcet_agreement(
                rankings,
                utilities,
                ranked_methods,
                rated_methods,
                tiebreaker=tiebreaker,
            )
            for key, value in delta.items():
                out[key][n_cands] += value

    return out


def accumulate_spatial_sue_by_ncands(
    n_elections: int,
    *,
    n_voters: int,
    n_cands_list: Sequence[int],
    dims: int,
    corr: float,
    disp: float,
    ranked_methods: Mapping[str, RankedMethod],
    rated_methods: Mapping[str, RatedMethod],
    tiebreaker: str = "random",
) -> dict[str, Counter]:
    """
    Accumulate summed social utilities (plus random reference) by ``n_cands``.

    Uses :func:`spatial_random_reference_utility_updates` each election, matching
    Merrill figures 4.a / 4.b.
    """
    keys = ranked_methods.keys() | rated_methods.keys() | {"SU max", "RW"}
    utility_sums: dict[str, Counter] = {k: Counter() for k in keys}

    for _ in range(n_elections):
        for n_cands in n_cands_list:
            v, c = normal_electorate(n_voters, n_cands, dims=dims, corr=corr, disp=disp)
            utilities = normed_dist_utilities(v, c)
            rankings = honest_rankings(utilities)
            delta = spatial_random_reference_utility_updates(
                utilities,
                rankings,
                ranked_methods,
                rated_methods,
                tiebreaker=tiebreaker,
            )
            for name, value in delta.items():
                utility_sums[name][n_cands] += value

    return utility_sums
