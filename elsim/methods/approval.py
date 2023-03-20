import numpy as np
from elsim.methods._common import (_all_indices, _order_tiebreak_keep,
                                   _random_tiebreak, _no_tiebreak)

_tiebreak_map = {'order': _order_tiebreak_keep,
                 'random': _random_tiebreak,
                 None: _no_tiebreak}


def _get_tiebreak(tiebreaker):
    try:
        return _tiebreak_map[tiebreaker]
    except KeyError:
        raise ValueError('Tiebreaker not understood')


def approval(election, tiebreaker=None):
    """
    Find the winner of an election using approval voting.

    The candidate with the largest number of approvals wins. [1]_

    Parameters
    ----------
    election : array_like
        A 2D collection of approval ballots.

        Rows represent voters, and columns represent candidate IDs.
        A cell contains 1 if that voter approves of that candidate,
        otherwise 0.

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
    .. [1] https://en.wikipedia.org/wiki/Approval_voting

    Examples
    --------
    Voter 0 approves Candidate A (index 0) and B (index 1).
    Voter 1 approves B and C.
    Voter 2 approves B and C.

    >>> election = [[1, 1, 0],
    ...             [0, 1, 1],
    ...             [0, 1, 1],
    ...             ]

    Candidate B (1) gets the most approvals and wins the election:

    >>> approval(election)
    1

    """
    election = np.asarray(election, dtype=np.uint8)

    if election.min() < 0 or election.max() > 1:
        raise ValueError

    # Tally all approvals
    tallies = election.sum(axis=0)

    # Find the set of candidates who have the highest score (usually only one)
    highest = max(tallies)
    winners = _all_indices(tallies, highest)

    # Break any ties using specified method
    tiebreak = _get_tiebreak(tiebreaker)
    return tiebreak(winners)[0]


def combined_approval(election, tiebreaker=None):
    """
    Find the winner of an election using combined approval voting.

    Also known as balanced approval or dis&approval voting, the candidate with
    the largest number of approvals minus disapprovals wins. [1]_

    Parameters
    ----------
    election : array_like
        A 2D collection of combined approval ballots.

        Rows represent voters, and columns represent candidate IDs.
        A cell contains a +1 if that voter approves of that candidate, a -1 if
        the voter disapproves of that candidate, or a 0 if they are neutral.

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
    .. [1] https://en.wikipedia.org/wiki/Combined_approval_voting

    Examples
    --------
    Voter 0 approves Candidate A (index 0) and B (index 1).
    Voter 1 approves of B and disapproves of C.
    Voter 2 approves of A and disapproves of B and C.

    >>> election = [[+1, +1,  0],
    ...             [ 0, +1, -1],
    ...             [+1, -1, -1],
    ...             ]

    Candidate A (0) has the highest net approval and wins the election:

    >>> combined_approval(election)
    0

    """
    election = np.asarray(election, dtype=np.int8)

    if election.min() < -1 or election.max() > 1:
        raise ValueError

    # Tally all approvals
    tallies = election.sum(axis=0)

    # Find the set of candidates who have the highest score (usually only one)
    highest = max(tallies)
    winners = _all_indices(tallies, highest)

    # Break any ties using specified method
    tiebreak = _get_tiebreak(tiebreaker)
    return tiebreak(winners)[0]
