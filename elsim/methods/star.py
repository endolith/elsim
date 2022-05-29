import random
import numpy as np
from ._common import _all_indices


def _order_tiebreak(winners, n=1):
    """
    Given an iterable of possibly tied `winners`, select the lowest-numbered
    `n` candidates.  If `n` is larger than `winners`, it is returned unchanged.
    """
    return sorted(winners)[:n]


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


_tiebreak_map = {'order': _order_tiebreak,
                 'random': _random_tiebreak,
                 None: _no_tiebreak}


def _get_tiebreak(tiebreaker):
    try:
        return _tiebreak_map[tiebreaker]
    except KeyError:
        raise ValueError('Tiebreaker not understood')


def star(election, tiebreaker=None):
    """
    Find the winner of an election using STAR voting.

    The more-preferred of the two candidates with the highest scores wins.[1]_

    Parameters
    ----------
    election : array_like
        A 2D collection of score ballots.

        Rows represent voters, and columns represent candidate IDs.
        A cell contains a high score if that voter approves of that candidate,
        or low score if they disapprove

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
    .. [1] https://en.wikipedia.org/wiki/STAR_voting

    Examples
    --------
    Voter 0 loves Candidates A (index 0) and B (index 1), but hates C (2).
    Voter 1 dislikes A, likes B, and loves C.
    Voter 2 hates A, is lukewarm about B and likes C.

    >>> election = [[5, 5, 0],
                    [0, 4, 5],
                    [0, 3, 4]]

    Candidates B and C (1 and 2) get the highest scores (12 and 9).  Of these,
    C is preferred on more ballots, and wins the election:

    >>> star(election)
    2

    """
    election = np.asarray(election)

    if election.min() < 0:
        raise ValueError

    if election.shape[1] == 1:
        # Only 1 candidate: that candidate wins.
        return 0

    # Tally all scores
    tallies = election.sum(axis=0)

    # Find the set of candidates who have the highest score (usually only one)
    highest = max(tallies)
    first_set = _all_indices(tallies, highest)

    # TODO: Follow the official tie-breaking rules
    # Break any ties using specified method
    tiebreak = _get_tiebreak(tiebreaker)

    if len(first_set) == 2:
        first, second = first_set
    elif len(first_set) > 2:
        result = tiebreak(first_set, 2)
        if result[0] is None:
            return None
        else:
            first, second = result
    elif len(first_set) == 1:
        first = first_set[0]
        # Find the set of candidates who have the second-highest score
        second = np.sort(tallies)[-2]
        second_set = _all_indices(tallies, second)
        if len(second_set) == 1:
            second = second_set[0]
        else:
            second = tiebreak(second_set, 1)[0]
        if second is None:
            return None
    else:
        raise RuntimeError('This should not happen')

    # TODO there must be a vectorized way to do this
    a = first
    b = second
    a_beats = 0
    b_beats = 0
    # Find the candidate who was preferred on more ballots
    for ballot in election:
        if ballot[a] > ballot[b]:
            a_beats += 1
        elif ballot[a] < ballot[b]:
            b_beats += 1

    if a_beats > b_beats:
        winner = a
    elif a_beats < b_beats:
        winner = b
    else:
        winner = tiebreak({first, second}, 1)[0]

    return winner
