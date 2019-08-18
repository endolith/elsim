import numpy as np
from random import choice


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
    first_preferences = election[:, 0]
    tallies = np.bincount(first_preferences).tolist()
    highest = max(tallies)
    if tiebreaker == 'order':
        # Returns first instance in candidate order
        return tallies.index(highest)
    elif tiebreaker == 'random':
        winners = [i for i, x in enumerate(tallies) if x == highest]
        return choice(winners)
    elif tiebreaker is None:
        n_winners = tallies.count(highest)
        if n_winners == 1:
            return tallies.index(highest)
        elif n_winners > 1:
            # There is a tie
            return None
        else:
            raise RuntimeError('Bug in FPTP')
    else:
        raise ValueError('Tiebreaker not understood')


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
