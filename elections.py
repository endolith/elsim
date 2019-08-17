import numpy as np

"""
Originally used shuffle but it was much slower:

    ordered_ranking = np.arange(n_cands, dtype=np.uint8)  # 1 μs
    rankings = np.tile(ordered_ranking, (n_voters, 1))  # 42 μs
    for ndx in range(n_voters):  # 300 ms
        np.random.shuffle(rankings[ndx])  # 29 μs each
    return rankings

Also tried disarrange(), permutations(), etc. All take about the same amount of
time.

Merrill 1984 uses this utility method and it's much faster, so I will use it,
too.  Raw random utilities are necessary for other tests, anyway.
"""


def impartial_culture(n_voters, n_cands):
    """
    Generates ranked ballots using the impartial culture / random society model

    The impartial culture model selects complete preference rankings from the
    set of all possible preference rankings using a uniform distribution.

    First a set of independent, uniformly distributed random utilities are
    generated, then these are converted into rankings.

    Parameters
    ----------
    n_voters : int
        Number of voters
    n_cands : int
        Number of candidates

    Returns
    -------
    election : numpy.ndarray
        A collection of ranked ballots.
        Rows represent voters and columns represent rankings, from best to
        worst, with no tied rankings.
        Each cell contains the ID number of a candidate, starting at 0.

        For example, if a voter ranks Curie > Avogadro > Bohr, the ballot line
        would read ``[2, 0, 1]`` (with IDs in alphabetical order).
    """
    if n_cands > 256:
        raise ValueError('Maximum number of candidates is 256')
    utilities = np.random.rand(n_voters, n_cands)

    # Technically this is upside-down, but it doesn't make any difference
    # uint8 because it doesn't need to support more than 256 candidates
    rankings = np.argsort(utilities).astype(np.uint8)
    return rankings


if __name__ == "__main__":
    # Run unit tests, in separate process to avoid warnings about cached
    # modules, printing output line by line in realtime
    from subprocess import Popen, PIPE
    with Popen(['pytest',
                '--tb=short',  # shorter traceback format
                '--hypothesis-show-statistics',
                str('test_elections.py')], stdout=PIPE, bufsize=1,
               universal_newlines=True) as p:
        for line in p.stdout:
            print(line, end='')
