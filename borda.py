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


def borda(election, tiebreaker=None):
    """
    Finds the winner of a ranked ballot election using the Borda count method

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


if __name__ == "__main__":
    # Run unit tests, in separate process to avoid warnings about cached
    # modules, printing output line by line in realtime
    from subprocess import Popen, PIPE
    with Popen(['pytest',
                '--tb=short',  # shorter traceback format
                '--hypothesis-show-statistics',
                str('test_borda.py')], stdout=PIPE, bufsize=1,
               universal_newlines=True) as p:
        for line in p.stdout:
            print(line, end='')
