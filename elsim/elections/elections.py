import numpy as np
from scipy.spatial.distance import cdist
from elsim.strategies import honest_rankings


def random_utilities(n_voters, n_cands):
    """
    Generate utilities using the impartial culture / random society model.

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
        A collection of utilities between 0 and 1, inclusive.
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


def impartial_culture(n_voters, n_cands):
    """
    Generate ranked ballots using the impartial culture / random society model.

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
    rankings = honest_rankings(utilities)
    return rankings


def normal_electorate(n_voters, n_cands, dims=2, corr=0.0, disp=1.0):
    """
    Generate normally distributed voters and candidates in issue space.

    Parameters
    ----------
    n_voters : int
        Number of voters
    n_cands : int
        Number of candidates
    dims : int
         Number of dimensions
    corr : float
        Correlation between each pair of random variables
    disp : float
        The relative dispersions of voters vs candidates, as a ratio of
        standard deviations.  For example, 1.0 means they are distributed by
        the same amount, while 0.5 means that candidates are more tightly
        concentrated than voters.

    Returns
    -------
    voters : numpy.ndarray
        Positions of voters in N-dimensional space, of shape
        ``(n_voters, n_dimensions)``.
    cands : numpy.ndarray
        Positions of candidates in N-dimensional space, of shape
        ``(n_candidates, n_dimensions)``.

    Notes
    -----
    For computational efficiency, this behaves as if it had first generated an
    n×n correlation matrix with `corr` correlation between each pair of
    variables, then diagonalized it, rotating it to principal axes, so the
    variables became uncorrelated (though the distribution kept the same
    shape), then generated independent normally distributed variables from it.

    For 4 dimensions and a correlation of C, for instance, the correlation
    matrix is::

        [[1, C, C, C],
         [C, 1, C, C],
         [C, C, 1, C],
         [C, C, C, 1]]

    This can also be interpreted as a covariance matrix, since only relative
    variance matters, not absolute variance.

    After rotating the distribution, the covariance matrix becomes::

        [[A, 0, 0, 0],
         [0, B, 0, 0],
         [0, 0, B, 0],
         [0, 0, 0, B]]

    where ``B = 1 - corr`` and ``A = 1 + (dims - 1)*corr``

    And to simplify even more, the first variable becomes scaled by A/B, while
    the rest are 0 or 1, since again, only the relative variance matters.

    References
    ----------
    .. [1] S. Merrill III, "A Comparison of Efficiency of Multicandidate
           Electoral Systems", American Journal of Political Science, vol. 28,
           no. 1, p. 26, 1984.  :doi:`10.2307/2110786`
    """
    A = 1 + (dims - 1)*corr
    B = 1 - corr

    # Correlation is proportional to variance, while raw values are
    # proportional to standard deviation, and SD = √(variance)
    scale = np.sqrt(A/B)

    voters = np.random.standard_normal((n_voters, dims))
    voters[:, 0] *= scale

    candidates = np.random.standard_normal((n_cands, dims))
    candidates[:, 0] *= scale

    # Scale all coordinates relative to voter distribution
    candidates *= disp

    return voters, candidates


def normed_dist_utilities(voters, cands):
    """
    Generate normalized utilities from a spatial model.

    Given the positions of voters and candidates, calculate the distance from
    each voter to each candidate, and then assign proportional utilities, where
    the farthest candidate from each voter has a utility of 0 and the nearest
    has a utility of 1.

    Parameters
    ----------
    voters : array_like
        Positions of voters in N-dimensional space, of shape
        ``(n_voters, n_dimensions)``.
    cands : array_like
        Positions of candidates in N-dimensional space, of shape
        ``(n_candidates, n_dimensions)``.

    Returns
    -------
    election : numpy.ndarray
        A collection of utilities between 0 and 1, inclusive.
        Rows represent voters and columns represent candidates.

    References
    ----------
    .. [1] S. Merrill III, "A Comparison of Efficiency of Multicandidate
           Electoral Systems", American Journal of Political Science, vol. 28,
           no. 1, p. 26, 1984.  :doi:`10.2307/2110786`

    Examples
    --------
    Given an election with 3 voters and 3 candidates in a 2D space:

    >>> voters = ((1, 1),
                  (6, 3),
                  (1, 7))
    >>> cands = ((2, 3),
                 (5, 1),
                 (4, 6))

    Calculate their normalized utilities:

    >>> normed_dist_utilities(voters, cands)
    array([[1.   , 0.509, 0.   ],
           [0.   , 1.   , 0.224],
           [0.763, 0.   , 1.   ]])
    """
    # Find ideological distance between each voter and each candidate
    dists = cdist(voters, cands)

    # When distance is low, utility is high
    # "u(d) = - d, where d is the distance from voter to candidate."
    utilities = -dists

    # Normalize utilities to [0, 1] for each voter
    utilities -= utilities.min(1)[:, np.newaxis]
    utilities /= utilities.max(1)[:, np.newaxis]

    # TODO: Actually Merrill says: "For the spatial-model simulations standard
    # scores were used as the most practical computational method."

    return utilities
