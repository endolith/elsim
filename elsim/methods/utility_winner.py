import numpy as np
from ._common import (_all_indices, _order_tiebreak_keep, _random_tiebreak,
                      _no_tiebreak)

_tiebreak_map = {'order': _order_tiebreak_keep,
                 'random': _random_tiebreak,
                 None: _no_tiebreak}


def _get_tiebreak(tiebreaker):
    try:
        return _tiebreak_map[tiebreaker]
    except KeyError:
        raise ValueError('Tiebreaker not understood')


def utility_winner(utilities, tiebreaker=None):
    """
    Find the utilitarian winner of an election. (Dummy "election method").

    Given a set of utilities for each voter-candidate pair, find the candidate
    who maximizes summed utility across all voters.[1]_

    Parameters
    ----------
    utilities : array_like
        A 2D collection of utilities.

        Rows represent voters, and columns represent candidate IDs.
        Higher utility numbers mean greater approval of that candidate by that
        voter.

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
    .. [1] S. Merrill III, "A Comparison of Efficiency of Multicandidate
       Electoral Systems", American Journal of Political Science, vol. 28,
       no. 1, p. 41, 1984.  :doi:`10.2307/2110786`

    Examples
    --------
    Voter 0 loves Candidates A (index 0) and B (index 1), but hates C (2).
    Voter 1 dislikes A, likes B, and loves C.
    Voter 2 hates A, and is lukewarm about B and C.

    >>> utilities = [[1.0, 1.0, 0.0],
                     [0.1, 0.8, 1.0],
                     [0.0, 0.5, 0.5],
                     ]

    Candidate B (1) has the highest overall support and is the utility winner:

    >>> utility_winner(election)
    1
    """
    utilities = np.asarray(utilities)

    # Tally all utilities
    total_utilities = utilities.sum(0).tolist()

    # Find the set of candidates who have the highest score (usually only one)
    highest = max(total_utilities)
    winners = _all_indices(total_utilities, highest)

    # Break any ties using specified method
    tiebreak = _get_tiebreak(tiebreaker)
    return tiebreak(winners)[0]
