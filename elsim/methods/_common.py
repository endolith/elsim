"""
The first solutions I tried, such as np.unique(return_counts=True), were very
slow, so I made this numba solution, but then discovered np.add.at, which is
faster than the previous solutions, but still slower than numba.  So I'm using
numba as a "soft" dependency, falling back on numpy if not installed.
"""
import warnings
import random
import numpy as np

try:
    from numba import njit, NumbaPendingDeprecationWarning

    # Reflected set will be replaced in numba 0.46 and removed in 0.47.
    warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)

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
def _tally_at_pointer(tallies, election, pointer):
    """
    Tally candidates at the location pointed to, re-using tallies array.
    """
    # Clear tally array
    tallies[:] = 0
    n_voters = election.shape[0]
    for voter in range(n_voters):
        cand = election[voter, pointer[voter]]
        tallies[cand] += 1


@njit(cache=True, nogil=True)
def _inc_pointer(election, pointer, eliminated):
    """
    Update pointer to point at candidates that haven't been eliminated.
    """
    n_voters = election.shape[0]
    for voter in range(n_voters):
        while election[voter, pointer[voter]] in eliminated:
            pointer[voter] += 1


@njit(cache=True, nogil=True)
def _dec_pointer(election, pointer, eliminated):
    """
    Update pointer to point at candidates that haven't been eliminated.
    """
    n_voters = election.shape[0]
    for voter in range(n_voters):
        while election[voter, pointer[voter]] in eliminated:
            pointer[voter] -= 1


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
    try:
        return tiebreak_map[tiebreaker]
    except KeyError:
        raise ValueError(f'Tiebreaker {tiebreaker} not understood')