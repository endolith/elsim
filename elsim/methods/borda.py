import random
import numpy as np
from ._common import _all_indices


def _order_tiebreak(winners):
    """
    Given an iterable of possibly tied `winners`, select the lowest numbered.
    """
    return min(winners)


def _random_tiebreak(winners):
    """
    Given an iterable of possibly tied `winners`, select one at random.
    """
    return random.choice(winners)


def _no_tiebreak(winners):
    """
    Given an iterable of `winners`, return None if there is a tie.
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


def borda(election, tiebreaker=None):
    """
    Find the winner of a ranked ballot election using the Borda count method.

    A voter's lowest-ranked candidate receives 1 point, second-lowest receives
    2 points, and so on.  All points are summed, and the highest-scoring
    candidate wins.[1]_

    Parameters
    ----------
    election : array_like
        A collection of ranked ballots.
        Rows represent voters and columns represent rankings, from best to
        worst, with no tied rankings.
        Each cell contains the ID number of a candidate, starting at 0.

        For example, if a voter ranks Curie > Avogadro > Bohr, the ballot line
        would read ``[2, 0, 1]`` (with IDs in alphabetical order).
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
    .. [1] https://en.wikipedia.org/wiki/Borda_count

    Examples
    --------
    Label some candidates:

    >>> A, B, C = 0, 1, 2

    Specify the ballots for the 5 voters:

    >>> election = [[A, C, B],
                    [A, C, B],
                    [B, C, A],
                    [B, C, A],
                    [C, A, B],
                    ]

    Candidate A gets a total of 3+3+1+1+2 = 10 points.
    Candidate B gets a total of 1+1+3+3+1 =  9 points.
    Candidate C gets a total of 2+2+2+2+3 = 11 points.
    Candidate C is the winner:

    >>> borda(election)
    2

    """
    election = np.asarray(election)

    ncands = election.shape[1]
    total_tally = np.zeros(ncands, dtype=int)

    # Tally candidates in each column, multiply by points for each rank level
    for n, column in enumerate(election.T):
        tally = np.bincount(column, minlength=ncands)
        total_tally += (ncands - n)*tally

    # Python lists are faster than NumPy here
    total_tally = total_tally.tolist()

    # Find the set of candidates who have the highest score (usually only one)
    winners = _all_indices(total_tally, max(total_tally))

    # Break any ties using specified method
    tiebreak = _get_tiebreak(tiebreaker)
    return tiebreak(winners)
