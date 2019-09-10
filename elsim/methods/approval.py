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


def approval(election, tiebreaker=None):
    """
    Finds the winner of an election using approval voting

    The candidate with the largest number of approvals wins.[1]_

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
                    [0, 1, 1],
                    [0, 1, 1],
                    ]

    Candidate B (1) gets the most approvals and wins the election:

    >>> approval(election)
    1

    """
    election = np.asarray(election, dtype=np.uint8)

    # Tally all approvals
    tallies = election.sum(0)

    # Find the set of candidates who have the highest score (usually only one)
    winners = _all_indices(tallies, max(tallies))

    # Break any ties using specified method
    tiebreak = _get_tiebreak(tiebreaker)
    return tiebreak(winners)


def approval_optimal(utilities, tiebreaker=None):
    """
    Finds the winner of an election using optimal approval voting strategy

    Given a set of utilities for each voter-candidate pair, each voter is
    modeled as maximizing their expected utility, by approving any candidate
    that exceeds their mean utility over all candidates.[1]_

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
       no. 1, p. 26, 1984.  :doi:`10.2307/2110786`

    Examples
    --------
    Voter 0 loves Candidates A (index 0) and B (index 1), but hates C (2).
    Voter 1 dislikes A, likes B, and loves C.
    Voter 2 hates A, and is lukewarm about B and C.

    >>> utilities = [[1.0, 1.0, 0.0],
                     [0.1, 0.8, 1.0],
                     [0.0, 0.5, 0.5],
                     ]

    Each voter optimally chooses their approval threshold based on their mean
    utility:
    Voter 0 approves A and B.
    Voter 1 approves B and C.
    Voter 2 approves B and C.

    Candidate B (1) gets the most approvals and wins the election:

    >>> approval_optimal(utilities)
    1
    """
    means = np.mean(utilities, 1)
    approvals = (utilities > means[:, np.newaxis]).astype(np.uint8)
    return approval(approvals, tiebreaker)


if __name__ == "__main__":
    # Run unit tests, in separate process to avoid warnings about cached
    # modules, printing output line by line in realtime
    from subprocess import Popen, PIPE
    with Popen(['pytest',
                '--tb=short',  # shorter traceback format
                '--hypothesis-show-statistics',
                str('test_approval.py')], stdout=PIPE, bufsize=1,
               universal_newlines=True) as p:
        for line in p.stdout:
            print(line, end='')
