import random
import numpy as np


# https://stackoverflow.com/a/6294205/125507
def _all_indices(iterable, value):
    """
    Return all indices of `iterable` that match `value`.
    """
    return [i for i, x in enumerate(iterable) if x == value]


def _order_tiebreak(winners):
    """
    Given an iterable of possibly tied `winners`, select the lowest numbered
    """
    return min(winners)


def _random_tiebreak(winners):
    """
    Given an iterable of possibly tied `winners`, select one at random
    """
    return random.choice(winners)


def _no_tiebreak(winners):
    """
    Given an iterable of possibly tied `winners`, return None if there is a tie
    """
    if len(winners) == 1:
        return winners[0]
    else:
        return None


_tiebreak_map = {'order': _order_tiebreak,
                 'random': _random_tiebreak,
                 None: _no_tiebreak}


def _get_tiebreak(tiebreaker):
    try:
        return _tiebreak_map[tiebreaker]
    except KeyError:
        raise ValueError('Tiebreaker not understood')


def fptp(election, tiebreaker=None):
    """
    Finds the winner of an election using first-past-the-post / plurality rule

    The candidate with the largest number of first preferences wins.[1]_

    Parameters
    ----------
    election : array_like
        A 2D collection of ranked ballots.  (See `borda` for election format.)
        Or a 1D array of first preferences only.
    tiebreaker : {'random', 'order', None}, optional
        If there is a tie, and `tiebreaker` is ``'random'``, a random finalist
        is returned.
        If 'order', the lowest-ID tied candidate is returned.
        By default, ``None`` is returned for ties.

    Returns
    -------
    winner : int
        The ID number of the winner, or ``None`` for an unbroken tie.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Plurality_voting

        Examples
    --------
    Label some candidates:

    >>> A, B, C = 0, 1, 2

    Specify the ballots for the 5 voters:

    >>> election = [[A, C, B],
                    [A, C, B],
                    [B, A, C],
                    [B, C, A],
                    [B, C, A],
                    [C, A, B],
                    ]

    Candidate B (1) gets the most first-preference votes, and is the winner:

    >>> fptp(election)
    1

    Single-mark ballots can also be tallied (with ties broken as specified):

    >>> election = [A, B, B, C, C]
    >>> print(fptp(election))
    None

    There is a tie between B (1) and C (2).  ``tiebreaker=order`` always
    prefers the lower-numbered candidate in a tie:

    >>> fptp(election, 'order')
    1
    """
    election = np.asarray(election)

    # Tally all first preferences (with index of tally = candidate ID)
    if election.ndim == 2:
        first_preferences = election[:, 0]
    elif election.ndim == 1:
        first_preferences = election
    else:
        raise ValueError('Election array must be 2D ranked ballots or 1D'
                         'list of first preferences')
    tallies = np.bincount(first_preferences).tolist()

    # Find the set of candidates who have the highest score (usually only one)
    winners = _all_indices(tallies, max(tallies))

    # Break any ties using specified method
    tiebreak = _get_tiebreak(tiebreaker)
    return tiebreak(winners)
