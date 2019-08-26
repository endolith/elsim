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


def _count(election, eliminated):
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


def irv(election, tiebreaker=None):
    """
    Finds the winner of an election using instant-runoff voting

    If any candidate gets a majority of first-preference votes, they win.
    Otherwise, the candidate(s) with the least number of votes is eliminated,
    votes for eliminated candidates are transferred according to the voters'
    preference rankings, and a series of runoff elections are held between the
    remainders until a candidate gets a majority.[1]_

    Also known as "the alternative vote", "ranked-choice voting", Hare's
    method, or Ware's method.

    The votes in each instant-runoff round are calculated from the same set of
    ranked ballots.  If voters are honest and consistent between rounds, then
    this is also equivalent to the exhaustive ballot method, which uses actual
    separate runoff elections.[2]_

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
    .. [1] https://en.wikipedia.org/wiki/Instant-runoff_voting
    .. [2] https://en.wikipedia.org/wiki/Exhaustive_ballot
    """
    election = np.asarray(election)
    n_voters = election.shape[0]
    n_cands = election.shape[1]
    eliminated = list()
    tiebreak = _get_tiebreak(tiebreaker)

    for iteration in range(n_cands):
        tallies = _count(election, eliminated).tolist()

        # Did anyone get majority
        highest = max(tallies)
        if highest > n_voters / 2:
            return tallies.index(highest)

        # If not, eliminate lowest
    #    lowest = min(tallies[np.nonzero(tallies)])  # slower?
        lowest = min(x for x in tallies if x != 0)  # faster?
        low_scorers = _all_indices(tallies, lowest)

        loser = tiebreak(low_scorers)[0]
        # Handle no tiebreaker case
        if loser is None:
            return None
        eliminated.append(loser)
    raise RuntimeError('Bug in IRV calculation')


if __name__ == "__main__":
    # Run unit tests, in separate process to avoid warnings about cached
    # modules, printing output line by line in realtime
    from subprocess import Popen, PIPE
    with Popen(['pytest',
                '--tb=short',  # shorter traceback format
                '--hypothesis-show-statistics',
                str('test_irv.py')], stdout=PIPE, bufsize=1,
               universal_newlines=True) as p:
        for line in p.stdout:
            print(line, end='')
