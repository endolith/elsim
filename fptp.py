import numpy as np


def fptp(election, tiebreaker=None):
    """
    Finds the winner of an election using first-past-the-post/plurality

    Parameters
    ----------
    election : array_like
        A collection of ranked ballots.  See `borda` for election format.
    tiebreaker : {'random', None}, optional
        If there is a tie, and `tiebreaker` is ``'random'``, a random finalist
        is returned.  By default, ``None`` is returned for ties.

    Returns
    -------
    winner : int
        The ID number of the winner, or ``None`` for an unbroken tie.
    """
    # TODO: It should also accept a 1D array of first preferences
    election = np.asarray(election)
    first_preferences = election[:, 0]
    tallies = np.bincount(first_preferences)
    highest_tally = tallies.max()
    if tiebreaker == 'random':
        return np.nonzero(tallies == highest_tally)[0][0]
    elif tiebreaker is None:
        # TODO: bincount should not be used for tallies, it could be huge
        n_winners = np.bincount(tallies)[highest_tally]
        if n_winners == 1:
            return np.nonzero(tallies == highest_tally)[0][0]
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
