import numpy as np
from elsim.methods._common import (_all_indices, _get_tiebreak, _no_tiebreak,
                                   _order_tiebreak_keep, _random_tiebreak)

_tiebreak_map = {'order': _order_tiebreak_keep,
                 'random': _random_tiebreak,
                 None: _no_tiebreak}


def score(election, tiebreaker=None):
    """
    Find the winner of an election using score (or range) voting.

    The candidate with the highest score wins. [1]_

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
    ...             [0, 4, 5],
    ...             [0, 3, 3]]

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
    highest = max(tallies)
    winners = _all_indices(tallies, highest)

    # Break any ties using specified method
    tiebreak = _get_tiebreak(tiebreaker, _tiebreak_map)
    return tiebreak(winners)[0]
