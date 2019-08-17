from condorcet import condorcet
from borda import borda


def black(election):
    """
    Finds the winner of a ranked ballot election using Black's method

    If a Condorcet winner exists, it is returned, otherwise, the Borda winner
    is returned.

    Parameters
    ----------
    election : array_like
        A collection of ranked ballots.
        Rows represent voters and columns represent rankings, from best to
        worst, with no tied rankings.
        Each cell contains the ID number of a candidate, starting at 0.

        For example, if a voter ranks Curie > Avogadro > Bohr, the ballot line
        would read ``[2, 0, 1]`` (with IDs in alphabetical order).

    Returns
    -------
    winner : int
        The ID number of the winner, or ``None`` for a Borda count tie.
    """
    winner = condorcet(election)
    if winner is None:
        winner = borda(election)
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
