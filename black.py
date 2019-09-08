from condorcet import condorcet
from borda import borda


def black(election, tiebreaker=None):
    """
    Finds the winner of a ranked ballot election using Black's method

    If a Condorcet winner exists, it is returned, otherwise, the Borda winner
    is returned.[1]_

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
        If there is a tie in the Borda tally, and `tiebreaker` is ``'random'``,
        a random finalist is returned.
        If 'order', the lowest-ID tied candidate is returned.
        By default, ``None`` is returned for ties.

    Returns
    -------
    winner : int
        The ID number of the winner, or ``None`` for a Borda count tie.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Condorcet_method#Two-method_systems

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

    A is preferred over B by 3 voters.
    C is preferred over A by 3 voters.
    C is preferred over B by 3 voters.
    C is thus the Condorcet winner and wins under Black's method:

    >>> black(election)
    2
    """
    winner = condorcet(election)
    if winner is None:
        winner = borda(election, tiebreaker)
    return winner


if __name__ == "__main__":
    # Run unit tests, in separate process to avoid warnings about cached
    # modules, printing output line by line in realtime
    from subprocess import Popen, PIPE
    with Popen(['pytest',
                '--tb=short',  # shorter traceback format
                '--hypothesis-show-statistics',
                str('test_black.py')], stdout=PIPE, bufsize=1,
               universal_newlines=True) as p:
        for line in p.stdout:
            print(line, end='')
