"""
Internal helpers for :mod:`elsim.methods`.
"""
import random
import warnings

import numpy as np

try:
    from numba import njit

    numba_enabled = True
except ImportError:
    warnings.warn('Numba not installed, Condorcet code will run slower')

    def njit(*args, **kwargs):
        """
        Do-nothing dummy decorator for when numba not installed.
        """
        def decorator(func):
            return func
        return decorator

    numba_enabled = False


# The first solutions I tried, such as np.unique(return_counts=True), were very
# slow, so I made this numba solution, but then discovered np.add.at, which is
# faster than the previous solutions, but still slower than numba.  So I'm
# using numba as a "soft" dependency, falling back on numpy if not installed.
if numba_enabled:
    @njit(cache=True, nogil=True)
    def _tally_pairs(pairs, tally):
        """
        Takes a 3D array of pairs and a 2D tallying array and modifies the
        tallying array in-place to tally the pairs
        """
        for i in range(pairs.shape[0]):
            for j in range(pairs.shape[1]):
                pair = pairs[i][j]
                tally[pair[0], pair[1]] += 1
else:
    def _tally_pairs(pairs, tally):
        """
        Takes a 3D array of pairs and a 2D tallying array and modifies the
        tallying array in-place to tally the pairs
        """
        np.add.at(tally, tuple(pairs.T), 1)


@njit(cache=True, nogil=True)
def _tally_at_rank_idx(tallies, election, rank_idx):
    """
    Tally the candidate at each voter's current rank index.

    Clears ``tallies`` then, for each voter, increments the count for the
    candidate at ``election[voter, rank_idx[voter]]``.

    Parameters
    ----------
    tallies : ndarray
        1D array of length ``n_cands``, modified in place. Is reset to zero
        before tallying.
    election : ndarray
        Ranked ballots: rows are voters, columns are rank positions from best
        to worst; ``election[i, j]`` is the candidate ID voter ``i`` assigns to
        rank ``j``. Shape ``(n_voters, n_ranks)`` (often ``n_ranks`` equals the
        number of candidates when ballots are complete).
    rank_idx : ndarray
        1D array of length ``n_voters``. Per-voter column index into
        ``election`` for the rank being tallied (e.g. top after eliminations in
        IRV, or bottom for Coombs last-place counts), depending on the caller.
    """
    # Clear tally array
    tallies[:] = 0
    n_voters = election.shape[0]
    for voter in range(n_voters):
        cand = election[voter, rank_idx[voter]]
        tallies[cand] += 1


@njit(cache=True, nogil=True)
def _inc_rank_idx(election, rank_idx, eliminated_mask):
    """
    Advance each voter's top-rank cursor past eliminated candidates.

    For each voter, increases ``rank_idx[voter]`` until the candidate at
    ``election[voter, rank_idx[voter]]`` is not eliminated.

    Parameters
    ----------
    election : ndarray
        Ranked ballots: rows are voters, columns are rank positions from best
        to worst; ``election[i, j]`` is the candidate ID at rank ``j``. Shape
        ``(n_voters, n_ranks)``.
    rank_idx : ndarray
        1D array of length ``n_voters``, modified in place. Column index for
        each voter's current first choice among remaining candidates.
    eliminated_mask : ndarray
        1D boolean array of length ``n_cands``. ``eliminated_mask[i]`` is True
        if candidate ``i`` is eliminated.
    """
    n_voters = election.shape[0]
    for voter in range(n_voters):
        while True:
            top_cand = election[voter, rank_idx[voter]]
            if eliminated_mask[top_cand]:
                rank_idx[voter] += 1
            else:
                break


@njit(cache=True, nogil=True)
def _dec_rank_idx(election, rank_idx, eliminated_mask):
    """
    Move each voter's bottom-rank cursor past eliminated candidates.

    For each voter, decreases ``rank_idx[voter]`` until the candidate at
    ``election[voter, rank_idx[voter]]`` is not eliminated.

    Parameters
    ----------
    election : ndarray
        Ranked ballots: rows are voters, columns are rank positions from best
        to worst; ``election[i, j]`` is the candidate ID at rank ``j``. Shape
        ``(n_voters, n_ranks)``.
    rank_idx : ndarray
        1D array of length ``n_voters``, modified in place. Column index for
        each voter's current last choice among remaining candidates (Coombs
        bottom scan).
    eliminated_mask : ndarray
        1D boolean array of length ``n_cands``. ``eliminated_mask[i]`` is True
        if candidate ``i`` is eliminated.
    """
    n_voters = election.shape[0]
    for voter in range(n_voters):
        while True:
            bottom_cand = election[voter, rank_idx[voter]]
            if eliminated_mask[bottom_cand]:
                rank_idx[voter] -= 1
            else:
                break


# https://stackoverflow.com/a/6294205/125507
def _all_indices(iterable, value):
    """
    Return all indices of `iterable` that match `value`.
    """
    return [i for i, x in enumerate(iterable) if x == value]


def _order_tiebreak_keep(winners, n=1):
    """
    Given an iterable of possibly tied `winners`, select the lowest-numbered
    `n` candidates.  If `n` is larger than `winners`, it is returned unchanged.
    """
    return sorted(winners)[:n]


def _order_tiebreak_elim(winners, n=1):
    """
    Given an iterable of possibly tied `winners`, select the highest numbered
    `n` candidates.  (Since they are to be eliminated, and all systems should
    be biased in favor of lower-indexed candidates.)  If `n` is larger than
    `winners`, it is returned unchanged.
    """
    return sorted(winners)[-n:]


def _random_tiebreak(winners, n=1):
    """
    Given an iterable of possibly tied `winners`, select `n` candidates at
    random.  If `n` is larger than `winners`, it is returned unchanged.
    """
    if len(winners) <= n:
        return winners
    else:
        return random.sample(winners, n)


def _no_tiebreak(winners, n=1):
    """
    Given an iterable of possibly tied `winners`, return None if there are more
    than `n` tied.  If `n` is larger than `winners`, it is returned unchanged.
    """
    if len(winners) <= n:
        return winners
    else:
        return [None]


def _get_tiebreak(tiebreaker, tiebreak_map):
    """
    Return the tiebreak callable for ``tiebreaker`` from ``tiebreak_map``.
    """
    try:
        return tiebreak_map[tiebreaker]
    except KeyError:
        raise ValueError(f'Tiebreaker {tiebreaker} not understood')
