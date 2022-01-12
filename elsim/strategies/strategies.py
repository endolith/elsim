import numpy as np


def honest_rankings(utilities):
    """
    Convert utilities into rankings using honest strategy.

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

    >>> utilities = np.array([[0.805, 0.759, 0.969],
                              [0.392, 0.416, 0.898],
                              [0.008, 0.702, 0.107],
                              [0.663, 0.575, 0.174]])
    >>> honest_rankings(utilities)
    array([[2, 0, 1],
           [2, 1, 0],
           [1, 2, 0],
           [0, 1, 2]], dtype=uint8)
    """
    n_cands = utilities.shape[1]

    # 255 candidates is plenty for real elections, so we'll limit it there and
    # use uint8 to save memory.
    if n_cands > 255:
        raise ValueError('Maximum number of candidates is 255')

    # Higher utilities for a voter are ranked first (earlier in row)
    return np.argsort(utilities)[:, ::-1].astype(np.uint8)


def honest_ratings(utilities, levels=5):
    """
    Convert utilities to optimal approval voting ballots.

    Given a set of utilities for each voter-candidate pair, each voter is
    modeled as maximizing their expected utility, by approving any candidate
    that exceeds their mean utility over all candidates.[1]_

    Parameters
    ----------
    utilities : array_like
        A 2D collection of utilities.

        Rows represent voters, and columns represent candidate IDs.
        Higher utility numbers mean greater approval of that candidate by that
        voter.

    Returns
    -------
    election : ndarray
        A 2D collection of approval ballots.

        Rows represent voters, and columns represent candidate IDs.
        A cell contains 1 if that voter approves of that candidate,
        otherwise 0.

    References
    ----------
    .. [1] S. Merrill III, "A Comparison of Efficiency of Multicandidate
       Electoral Systems", American Journal of Political Science, vol. 28,
       no. 1, p. 26, 1984.  :doi:`10.2307/2110786`

    Examples
    --------
    Voter 0 loves Candidates A (index 0) and B (index 1), but hates C (2).
    Voter 1 dislikes A, likes B, and loves C.
    Voter 2 hates A, and is lukewarm about B and C.

    >>> utilities = [[1.0, 1.0, 0.0],
                     [0.1, 0.8, 1.0],
                     [0.0, 0.5, 0.5],
                     ]

    Each voter optimally chooses their approval threshold based on their mean
    utility:
    Voter 0 approves A and B.
    Voter 1 approves B and C.
    Voter 2 approves B and C.

    >>> approval_optimal(utilities)
    array([[1, 1, 0],
           [0, 1, 1],
           [0, 1, 1]], dtype=uint8)

    """
    means = np.mean(utilities, 1)
    approvals = (utilities > means[:, np.newaxis]).astype(np.uint8)
    return approvals



def approval_optimal(utilities):
    """
    Convert utilities to optimal approval voting ballots.

    Given a set of utilities for each voter-candidate pair, each voter is
    modeled as maximizing their expected utility, by approving any candidate
    that exceeds their mean utility over all candidates.[1]_

    Parameters
    ----------
    utilities : array_like
        A 2D collection of utilities.

        Rows represent voters, and columns represent candidate IDs.
        Higher utility numbers mean greater approval of that candidate by that
        voter.

    Returns
    -------
    election : ndarray
        A 2D collection of approval ballots.

        Rows represent voters, and columns represent candidate IDs.
        A cell contains 1 if that voter approves of that candidate,
        otherwise 0.

    References
    ----------
    .. [1] S. Merrill III, "A Comparison of Efficiency of Multicandidate
       Electoral Systems", American Journal of Political Science, vol. 28,
       no. 1, p. 26, 1984.  :doi:`10.2307/2110786`

    Examples
    --------
    Voter 0 loves Candidates A (index 0) and B (index 1), but hates C (2).
    Voter 1 dislikes A, likes B, and loves C.
    Voter 2 hates A, and is lukewarm about B and C.

    >>> utilities = [[1.0, 1.0, 0.0],
                     [0.1, 0.8, 1.0],
                     [0.0, 0.5, 0.5],
                     ]

    Each voter optimally chooses their approval threshold based on their mean
    utility:
    Voter 0 approves A and B.
    Voter 1 approves B and C.
    Voter 2 approves B and C.

    >>> approval_optimal(utilities)
    array([[1, 1, 0],
           [0, 1, 1],
           [0, 1, 1]], dtype=uint8)

    """
    means = np.mean(utilities, 1)
    approvals = (utilities > means[:, np.newaxis]).astype(np.uint8)
    return approvals


def vote_for_k(utilities, k):
    """
    Convert utilities to approval voting ballots, approving top k candidates.

    Given a set of utilities for each voter-candidate pair, each voter is
    modeled as voting for the top `k` candidates.[1]_

    Parameters
    ----------
    utilities : array_like
        A 2D collection of utilities.

        Rows represent voters, and columns represent candidate IDs.
        Higher utility numbers mean greater approval of that candidate by that
        voter.
    k : int or 'half'
        The number of candidates approved of by each voter, or 'half' to make
        the number dependent on the number of candidates.  If a negative int,
        then vote for ``n - k`` candidates, where ``n`` is the total number.

    Returns
    -------
    election : ndarray
        A 2D collection of approval ballots.

        Rows represent voters, and columns represent candidate IDs.
        A cell contains 1 if that voter approves of that candidate,
        otherwise 0.

    References
    ----------
    .. [1] Weber, Robert J. (1978). "Comparison of Public Choice Systems".
       Cowles Foundation Discussion Papers. Cowles Foundation for Research in
       Economics. No. 498. https://cowles.yale.edu/publications/cfdp/cfdp-498

    Examples
    --------
    Voter 0 loves Candidates A (index 0) and B (index 1), but hates C (2).
    Voter 1 dislikes A, likes B, and loves C.
    Voter 2 hates A, and is lukewarm about B and C.

    >>> utilities = [[1.0, 1.0, 0.0],
                     [0.1, 0.8, 1.0],
                     [0.0, 0.5, 0.5],
                     ]

    Each voter approves of their top-two candidates:
    Voter 0 approves A and B.
    Voter 1 approves B and C.
    Voter 2 approves B and C.

    >>> vote_for_k(utilities, 2)
    array([[1, 1, 0],
           [0, 1, 1],
           [0, 1, 1]], dtype=uint8)

    """
    utilities = np.asarray(utilities)
    n_cands = utilities.shape[1]
    if k == 'half':
        # "It is interesting to observe that the vote-for-k and vote-for-(n-k)
        # voting systems are equally effective."
        # So for 7 candidates, we could use either k=4 or k=3 (= 7//2)
        # TODO: Though this seems only true with infinite voters?
        k = n_cands // 2
    elif -n_cands < k < 0:
        k = n_cands + k
    elif not 0 < k < n_cands:
        raise ValueError(f'k of {k} not possible with {n_cands} candidates')

    # Efficiently get indices of top k candidates for each voter
    # https://stackoverflow.com/a/23734295/125507
    # TODO: How are tied utilities handled, such as top 2 with 3 tied? Random?
    top_k = np.argpartition(utilities, -k, axis=1)[:, -k:]

    # Create blank ballots
    approvals = np.zeros(utilities.shape, np.uint8)

    # Fill in approvals
    # TODO: Not sure if this is the most efficient way
    approvals[np.arange(len(approvals))[:, np.newaxis], top_k] = 1
    return approvals
