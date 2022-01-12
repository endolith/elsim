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


def score(election, tiebreaker=None):
    """
    Find the winner of an election using score (or range) voting.

    The candidate with the highest score wins.[1]_

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
    .. [1] https://en.wikipedia.org/wiki/Score_voting

    Examples
    --------
    Voter 0 loves Candidates A (index 0) and B (index 1), but hates C (2).
    Voter 1 dislikes A, likes B, and loves C.
    Voter 2 hates A, and is lukewarm about B and C.

    >>> election = [[5, 5, 0],
                    [0, 4, 5],
                    [0, 3, 3]]

    Candidate B (1) gets the highest score and wins the election:

    >>> score(election)
    1

    """
    election = np.asarray(election)

    if election.min() < 0:
        raise ValueError

    # Tally all scores
    tallies = election.sum(axis=0)

    # Find the set of candidates who have the highest score (usually only one)
    winners = _all_indices(tallies, max(tallies))

    # Break any ties using specified method
    tiebreak = _get_tiebreak(tiebreaker)
    return tiebreak(winners)
