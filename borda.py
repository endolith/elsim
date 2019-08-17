import numpy as np


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
    tiebreaker : {'random', None}, optional
        If there is a tie, and `tiebreaker` is ``'random'``, a random finalist
        is returned.  By default, ``None`` is returned for ties.

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
    total_tally = list(total_tally)
    highest = max(total_tally)
    if tiebreaker == 'random':
        return total_tally.index(highest)
    elif tiebreaker is None:
        n_winners = total_tally.count(highest)
        if n_winners == 1:
            return total_tally.index(highest)
        elif n_winners > 1:
            # There is a tie
            return None
        else:
            raise RuntimeError('Bug in Borda count')
    else:
        raise ValueError('Tiebreaker not understood')


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
