import random
import numpy as np


# https://stackoverflow.com/a/6294205/125507
def _all_indices(iterable, value):
    """
    Return all indices of `iterable` that match `value`.
    """
    return [i for i, x in enumerate(iterable) if x == value]


def _order_tiebreak(winners, n=1):
    """
    Given an iterable of possibly tied `winners`, select the highest numbered
    (since they are to be eliminated)
    """
    return sorted(winners)[-n:]


def _random_tiebreak(winners, n=1):
    """
    Given an iterable of possibly tied `winners`, select one at random
    """
    if len(winners) == 1:
        return winners
    else:
        return random.sample(winners, n)


def _no_tiebreak(winners, n=1):
    """
    Given an iterable of possibly tied `winners`, return None if there is a tie
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


def _count_first(election, eliminated):
    """
    Count votes for each candidate who hasn't been eliminated
    """
    n_voters = election.shape[0]
    n_cands = election.shape[1]
    tallies = np.zeros(n_cands, dtype=int)
    counted = np.zeros(n_voters, dtype=bool)
    for column in election.T:

        # Count only votes for candidates who haven't been eliminated
        eligible = ~np.isin(column, eliminated)

        # Count candidates in this column who haven't been eliminated,
        # in rows that haven't yet been counted
        tallies += np.bincount(column[eligible & ~counted],
                               minlength=n_cands)

        # Keep track of which rows have already been counted
        counted += eligible

        # Quit early if all rows have been counted
        if np.all(counted):
            break

    return tallies


def _count_last(election, eliminated):
    """
    Count votes for each candidate who hasn't been eliminated
    """
    n_voters = election.shape[0]
    n_cands = election.shape[1]
    tallies = np.zeros(n_cands, dtype=int)
    counted = np.zeros(n_voters, dtype=bool)
    for column in election.T[::-1]:

        # Count only votes for candidates who haven't been eliminated
        eligible = ~np.isin(column, eliminated)

        # Count candidates in this column who haven't been eliminated,
        # in rows that haven't yet been counted
        tallies += np.bincount(column[eligible & ~counted],
                               minlength=n_cands)

        # Keep track of which rows have already been counted
        counted += eligible

        # Quit early if all rows have been counted
        if np.all(counted):
            break

    return tallies


def coombs(election, tiebreaker=None):
    """
    Finds the winner of an election using Coomb's method

    If any candidate gets a majority of first-preference votes, they win.
    Otherwise, the candidate(s) with the most number of last-preference votes
    is eliminated, votes for eliminated candidates are transferred according
    to the voters' preference rankings, and a series of runoff elections are
    held between the remainders until a candidate gets a majority.[1]_

    Parameters
    ----------
    election : array_like
        A collection of ranked ballots.  See `borda` for election format.
        Currently, this must include full rankings for each voter.
    tiebreaker : {'random', None}, optional
        If there is a tie, and `tiebreaker` is ``'random'``, tied candidates
        are eliminated or selected at random
        is returned.
        If 'order', the lowest-ID tied candidate is preferred in each tie.
        By default, ``None`` is returned if there are any ties.

    Returns
    -------
    winner : int
        The ID number of the winner, or ``None`` for an unbroken tie.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Coombs%27_method

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

    In the first round, no candidate gets a majority, so Candidate B (1) is
    eliminated, for receiving 3 out of 5 last-place votes.  Voter 2 and 3's
    support of B is transferred to Candidate C (2), causing Candidate C to win,
    with 3 out of 5 votes:

    >>> coombs(election)
    2
    """
    election = np.asarray(election)
    n_voters = election.shape[0]
    n_cands = election.shape[1]
    eliminated = list()
    tiebreak = _get_tiebreak(tiebreaker)

    for iteration in range(n_cands):
        tallies = _count_first(election, eliminated).tolist()

        # Did anyone get majority
        highest = max(tallies)
        if highest > n_voters / 2:
            return tallies.index(highest)

        # If not, eliminate candidate with highest number of last-preferences
        tallies = _count_last(election, eliminated).tolist()
        highest = max(tallies)
        highly_hated = _all_indices(tallies, highest)

        loser = tiebreak(highly_hated)[0]
        # Handle no tiebreaker case
        if loser is None:
            return None
        eliminated.append(loser)
    raise RuntimeError("Bug in Coombs' calculation")


if __name__ == "__main__":
    # Run unit tests, in separate process to avoid warnings about cached
    # modules, printing output line by line in realtime
    from subprocess import Popen, PIPE
    with Popen(['pytest',
                '--tb=short',  # shorter traceback format
                '--hypothesis-show-statistics',
                str('test_coombs.py')], stdout=PIPE, bufsize=1,
               universal_newlines=True) as p:
        for line in p.stdout:
            print(line, end='')
