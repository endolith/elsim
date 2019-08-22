import numpy as np


def random_utilities(n_voters, n_cands):
    """
    Generates utilities using the impartial culture / random society model

    The random society[1]_ or random uniform utilities[2]_ model selects
    independent candidate utilities for each voter from a uniform distribution
    in the interval [0, 1).

    This model is unrealistic, but is commonly used because it has some
    worst-case properties and is comparable between researchers.[3]_

    Parameters
    ----------
    n_voters : int
        Number of voters
    n_cands : int
        Number of candidates

    Returns
    -------
    election : numpy.ndarray
        A collection of utilities between 0 and 1.
        Rows represent voters and columns represent candidates.

    References
    ----------
    .. [1] S. Merrill III, "A Comparison of Efficiency of Multicandidate
           Electoral Systems", American Journal of Political Science, vol. 28,
           no. 1, p. 26, 1984.  :doi:`10.2307/2110786`
    .. [2] W.D. Smith, "Range voting", 2000,
           http://scorevoting.net/WarrenSmithPages/homepage/rangevote.pdf
    .. [3] A. Lehtinen and J. Kuorikoski, "Unrealistic Assumptions in Rational
           Choice Theory", Philosophy of the Social Sciences vol. 37, no. 2,
           p 132. 2007. :doi:`10.1177/0048393107299684`

    Examples
    --------
    Generate an election with 4 voters and 3 candidates:

    >>> random_utilities(4, 3)
    array([[0.805, 0.759, 0.969],
           [0.392, 0.416, 0.898],
           [0.008, 0.702, 0.107],
           [0.663, 0.575, 0.174]])

    Here, Voter 1 prefers Candidate 2, and considers Candidate 0 and 1 roughly
    similar.
    """
    # Generate utilities from a uniform distribution over [0, 1).
    # Merrill uses [0, 1], but that shouldn't make any difference.
    return np.random.rand(n_voters, n_cands)


def rankings_from_utilities(utilities):
    """
    Convert utilities into rankings

    Parameters
    ----------
    utilities : array_like
        A 2D collection of utilities.

        Rows represent voters, and columns represent candidate IDs.
        Higher utility numbers mean greater approval of that candidate by that
        voter.

    Returns
    -------
    election : array_like
        A collection of ranked ballots.
        Rows represent voters and columns represent rankings, from best to
        worst, with no tied rankings.
        Each cell contains the ID number of a candidate, starting at 0.

        For example, if a voter ranks Curie > Avogadro > Bohr, the ballot line
        would read ``[2, 0, 1]`` (with IDs in alphabetical order).

    Examples
    --------
    Generate an election with 4 voters and 3 candidates:

    >>> random_utilities(4, 3)
    array([[0.805, 0.759, 0.969],
           [0.392, 0.416, 0.898],
           [0.008, 0.702, 0.107],
           [0.663, 0.575, 0.174]])

    Here, Voter 2 prefers Candidate 1, then 2, then 0, as we can see when
    converted to rankings:

    >>> utilities = array([[0.805, 0.759, 0.969],
                           [0.392, 0.416, 0.898],
                           [0.008, 0.702, 0.107],
                           [0.663, 0.575, 0.174]])
    >>> rankings_from_utilities(utilities)
    array([[2, 0, 1],
           [2, 1, 0],
           [1, 2, 0],
           [0, 1, 2]], dtype=uint8)
    """
    # 256 candidates is plenty for real elections, so we'll limit it there and
    # use uint8 to save memory.
    n_cands = utilities.shape[1]
    if n_cands > 256:
        raise ValueError('Maximum number of candidates is 256')

    # Higher utilities for a voter are  ranked first (earlier in row)
    return np.argsort(utilities)[:, ::-1].astype(np.uint8)


def impartial_culture(n_voters, n_cands):
    """
    Generates ranked ballots using the impartial culture / random society model

    The impartial culture model selects complete preference rankings from the
    set of all possible preference rankings using a uniform distribution.

    This model is unrealistic, but is commonly used because it has some
    worst-case properties and is comparable between researchers.[2]_

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

    Notes
    -----
    This implementation first generates a set of independent, uniformly
    distributed random utilities, which are then converted into rankings.[1]_

    It can (extremely rarely) generate tied utilities, which are always ranked
    in order from lowest to highest, so there is a very slight bias in favor of
    lower-numbered candidates?

    References
    ----------
    .. [1] S. Merrill III, "A Comparison of Efficiency of Multicandidate
           Electoral Systems", American Journal of Political Science, vol. 28,
           no. 1, p. 26, 1984.  :doi:`10.2307/2110786`
    .. [2] A. Lehtinen and J. Kuorikoski, "Unrealistic Assumptions in Rational
           Choice Theory", Philosophy of the Social Sciences vol. 37, no. 2,
           p 132. 2007. :doi:`10.1177/0048393107299684`

    Examples
    --------
    Generate an election with 4 voters and 3 candidates:

    >>> impartial_culture(4, 3)
    array([[0, 1, 2],
           [2, 0, 1],
           [2, 1, 0],
           [1, 0, 2]], dtype=uint8)

    Here, Voter 1 prefers Candidate 2, then Candidate 0, then Candidate 1.
    """
    # This method is much faster than generating integer sequences and then
    # shuffling them.
    utilities = random_utilities(n_voters, n_cands)
    rankings = rankings_from_utilities(utilities)
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
