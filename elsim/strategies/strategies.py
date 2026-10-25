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
    ...                       [0.392, 0.416, 0.898],
    ...                       [0.008, 0.702, 0.107],
    ...                       [0.663, 0.575, 0.174]])
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


def honest_normed_scores(utilities, max_score=5):
    """
    Convert utilities into scores using honest (but normalized) strategy.

    Given a set of utilities for each voter-candidate pair, each voter is
    modeled as giving their favorite candidate a maximum score, least favorite
    candidate a minimum score, and proportional scores in between.

    Parameters
    ----------
    utilities : array_like
        A 2D collection of utilities.

        Rows represent voters, and columns represent candidate IDs.
        Higher utility numbers mean greater approval of that candidate by that
        voter.

    max_score : int, optional
        The highest score on the ballot. If `max_score` = 3, the possible
        scores are 0, 1, 2, 3.

    Returns
    -------
    election : ndarray
        A 2D collection of score ballots.

        Rows represent voters, and columns represent candidate IDs.
        A cell contains a high score if that voter approves of that candidate,
        or low score if they disapprove

    Examples
    --------
    Voter 0 loves Candidates A (index 0) and B (index 1), but hates C (2).
    Voter 1 dislikes A, likes B, and loves C.
    Voter 2 hates A, and is lukewarm about B and C.

    >>> utilities = [[1.0, 1.0, 0.0],
    ...              [0.1, 0.8, 1.0],
    ...              [0.0, 0.5, 0.5],
    ...              ]

    Each voter fills out a 0-5 Score ballot with their favorite and least
    favorite scored at the max and min:

    >>> honest_normed_scores(utilities, max_score=5)
    array([[5, 5, 0],
           [0, 4, 5],
           [0, 5, 5]], dtype=uint8)

    """
    # Slide every voter's minimum utility to 0
    normed = utilities - np.amin(utilities, axis=1)[:, np.newaxis]

    # If a ballot is all 0, suppress 0/0 warning.
    # astype(np.uint8) will convert NaN back to 0.
    with np.errstate(invalid='ignore'):
        # Normalize every voter's maximum utility to 1
        normed /= np.amax(normed, axis=1)[:, np.newaxis]

    # Normalize every voter's maximum score to max_score
    normed *= max_score

    # Quantize to discrete scale
    return np.around(normed).astype(np.uint8)


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
    ...              [0.1, 0.8, 1.0],
    ...              [0.0, 0.5, 0.5],
    ...              ]

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
    ...              [0.1, 0.8, 1.0],
    ...              [0.0, 0.5, 0.5],
    ...              ]

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


def vote_for_or_against_k(utilities, k):
    r"""
    Convert utilities to approval voting ballots, voting for top k or m-k.

    Weber's "vote for or against k" system offers every voter a choice between
    two ways of casting `k` approvals or `k` disapprovals: the weight sets
    ``(1, ..., 1, 0, ..., 0)`` with `k` ones, and ``(0, ..., 0, -1, ..., -1)``
    with `k` minus-ones.[1]_  Adding 1 to every weight of the second set does
    not change who wins, so the two options are simply "approve of my top `k`
    candidates" and "approve of my top `m - k` candidates", and that is what
    this function returns.

    Which option is optimal follows from Weber's optimal-strategy rule, which
    maximizes the sum of ``w_c * (u_c - ubar)`` over the allowed weight sets,
    where ``ubar`` is the voter's average utility.  Writing ``u_1 >= ... >=
    u_m`` for the voter's sorted utilities, the two options score

    .. math::

        G(k) = \sum_{i \leq k} (u_i - \bar u), \qquad
        G(m-k) = \sum_{i \leq m-k} (u_i - \bar u)

    and their difference is

    .. math::

        G(m-k) - G(k) = (m - 2k)\left(\bar u_{central} - \bar u\right),

    where :math:`\bar u_{central}` is the average utility of the ``m - 2k``
    centrally-ranked candidates.  So a voter approves of their top ``m - k``
    candidates if and only if their *central* candidates are, on average, above
    their average utility; otherwise they approve of only their top `k`
    candidates.  (When ``m = 2k`` the two options are the same ballot.)

    Parameters
    ----------
    utilities : array_like
        A 2D collection of utilities.

        Rows represent voters, and columns represent candidate IDs.
        Higher utility numbers mean greater approval of that candidate by that
        voter.
    k : int
        The number of candidates approved (if voting for k) or disapproved
        (if voting against k, i.e. voting for m-k).  Must be between 1 and
        ``m // 2``; the system is symmetric in `k` and ``m - k``, and for
        ``k > m/2`` the two weight sets would be the same.

    Returns
    -------
    election : ndarray
        A 2D collection of approval ballots.

        Rows represent voters, and columns represent candidate IDs.
        A cell contains 1 if that voter approves of that candidate,
        otherwise 0.

        These are ordinary approval ballots, so they can be tallied with
        `elsim.methods.approval`.  (A voter who chooses the "against" option
        casts ``0`` and ``-1`` rather than ``1`` and ``0`` in the paper's
        version, but that only shifts every candidate's total by the same
        amount, so the winner is the same; `elsim.methods.combined_approval`
        gives the same result if the "against" voters' ballots are shifted by
        -1.)

    References
    ----------
    .. [1] Weber, Robert J. (1978). "Comparison of Public Choice Systems".
       Cowles Foundation Discussion Papers. Cowles Foundation for Research in
       Economics. No. 498. https://cowles.yale.edu/publications/cfdp/cfdp-498

    Notes
    -----
    The effectiveness values published for this system (page 19 of [1]) are too
    high for ``m >= 5``; the closed form behind them is wrong, not this rule.
    See examples/weber_1977_expressions.py for the corrected values and why.

    Examples
    --------
    Voter 0 loves Candidates A (index 0) and B (index 1), but hates C, D, and
    E (2, 3, 4).
    Voter 1 dislikes A and E, likes B and D, and loves C.
    Voter 2 hates A, dislikes C and E, and is lukewarm about B and D.

    >>> utilities = [[1.0, 1.0, 0.0, 0.0, 0.0],
    ...              [0.1, 0.8, 1.0, 0.9, 0.2],
    ...              [0.0, 0.5, 0.1, 0.5, 0.1],
    ...              ]

    If `k` = 2, voters are required to vote for or against 2 candidates.

    Voter 0's central candidate (rank 2, 0-indexed) has utility 0.0.
    Mean utility is 0.4. 0.0 < 0.4, so vote for top 2 (A, B).

    Voter 1's central candidate has utility 0.8.
    Mean utility is 0.6. 0.8 >= 0.6, so vote for top 3 (B, C, D).

    Voter 2's central candidate has utility 0.1.
    Mean utility is 0.24. 0.1 < 0.24, so vote for top 2 (B, D).

    >>> vote_for_or_against_k(utilities, 2)
    array([[1, 1, 0, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 0, 1, 0]], dtype=uint8)

    """
    utilities = np.asarray(utilities)
    n_voters, n_cands = utilities.shape

    if not 0 < k <= n_cands // 2:
        raise ValueError(f'k={k} must be between 1 and m//2={n_cands // 2}')

    # Sorted utilities, best first: u_1 >= u_2 >= ... >= u_m
    sorted_utilities = np.sort(utilities, axis=1)[:, ::-1]

    if n_cands == 2 * k:
        # Both weight sets are the same ballot: approve the top k
        n_approved = np.full(n_voters, k)
    else:
        # The middle m-2k candidates decide it: if their average exceeds the
        # average of all candidates, approving the top m-k scores higher.
        mean_u = utilities.mean(axis=1)
        central_u = sorted_utilities[:, k:n_cands - k].mean(axis=1)
        n_approved = np.where(central_u >= mean_u, n_cands - k, k)

    # Rank the candidates for each voter (0 = most preferred), then approve
    # the first n_approved of them.  Note that argpartition can't be used here:
    # its kth argument is constant per call, and looping over the two possible
    # values of n_approved is slower than a single sort for these sizes.
    order = np.argsort(utilities, axis=1)[:, ::-1]
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order,
                      np.broadcast_to(np.arange(n_cands), order.shape), axis=1)
    return (ranks < n_approved[:, np.newaxis]).astype(np.uint8)
