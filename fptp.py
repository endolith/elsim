import numpy as np
import random


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
    Finds the winner of an election using first-past-the-post/plurality

    Parameters
    ----------
    election : array_like
        A collection of ranked ballots.  See `borda` for election format.
    tiebreaker : {'random', 'order', None}, optional
        If there is a tie, and `tiebreaker` is ``'random'``, a random finalist
        is returned.
        If 'order', the lowest-ID tied candidate is returned.
        By default, ``None`` is returned for ties.

    Returns
    -------
    winner : int
        The ID number of the winner, or ``None`` for an unbroken tie.
    """
    # TODO: It should also accept a 1D array of first preferences
    election = np.asarray(election)

    # Tally all first preferences (with index of tally = candidate ID)
    first_preferences = election[:, 0]
    tallies = np.bincount(first_preferences).tolist()

    # Find the set of candidates who have the highest score (usually only one)
    winners = _all_indices(tallies, max(tallies))

    # Break any ties using specified method
    tiebreak = _get_tiebreak(tiebreaker)
    return tiebreak(winners)


if __name__ == "__main__":
    # Run unit tests, in separate process to avoid warnings about cached
    # modules, printing output line by line in realtime
    from subprocess import Popen, PIPE
    with Popen(['pytest',
                '--tb=short',  # shorter traceback format
                '--hypothesis-show-statistics',
                str('test_fptp.py')], stdout=PIPE, bufsize=1,
               universal_newlines=True) as p:
        for line in p.stdout:
            print(line, end='')
