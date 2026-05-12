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

    >>> from elsim.elections import random_utilities
    >>> utilities = random_utilities(4, 3, random_state=1984)  # for doctest
    >>> utilities
    array([[0.19746307, 0.00903803, 0.78376658],
           [0.08090381, 0.50265116, 0.55887602],
           [0.74867306, 0.21977523, 0.12586929],
           [0.64267652, 0.15365841, 0.77633876]])

    Here, Voter 3 prefers Candidate 2, then 0, then 1, as we can see when
    converted to rankings:

    >>> honest_rankings(utilities)
    array([[2, 0, 1],
           [2, 1, 0],
           [0, 1, 2],
           [2, 0, 1]], dtype=uint8)
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

        scores = np.around(normed).astype(np.uint8)

    # Quantize to discrete scale
    return scores


def approval_optimal(utilities):
    """
    Convert utilities to optimal approval voting ballots.

    Given a set of utilities for each voter-candidate pair, each voter is
    modeled as maximizing their expected utility, by approving any candidate
    that exceeds their mean utility over all candidates. [1]_

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
    modeled as voting for the top `k` candidates. [1]_

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


def vote_for_or_against_k(utilities, k, rng=None):
    """
    Convert utilities to combined-approval ballots (vote-for-or-against-k).

    Weber (*Comparison of Public Choice Systems*, Cowles Discussion Paper 498)
    defines ``2 * (m choose k)`` strategic types: for each cardinality-``k`` set
    ``S``, types **vote for** ``S`` (``+1`` on ``S``) and **vote against** ``S``
    (``-1`` on ``S``), each with probability ``1 / (2 * (m choose k))``.  The
    effectiveness formulas follow from the resulting reproducing scores
    ``u_t(c)`` over regions of the preference cube. [1]_

    For **simulation**, each row uses ``utilities`` to rank candidates (with
    tie-breaking noise), assigns ``+1`` to the ``k`` **highest**-utility
    candidates and ``-1`` to the ``k`` **lowest** (zeros elsewhere).  When
    ``k <= n_cands // 2`` those sets are disjoint, so ballots stay in
    ``{-1, 0, +1}``.  With utilities drawn from impartial culture (e.g.
    :func:`~elsim.elections.random_utilities`), Monte Carlo Social Utility
    Efficiency matches Weber's Table 19 when tallies use combined approval, as
    in ``examples/weber_1977_effectiveness_table.py`` and ``eff_vote_for_or_against_k``.

    With i.i.d. continuous utilities, a voter's top-``k`` label set is
    **marginally** uniform over ``k``-subsets and the bottom-``k`` set is
    marginally uniform as well, but this function fixes both from one ranking,
    which differs from drawing a random ``k``-subset independently of
    ``utilities``; see :func:`vote_for_or_against_k_uniform_types`.

    Parameters
    ----------
    utilities : array_like
        A 2D collection of utilities.  Rows are voters; columns are candidates.
    k : int
        Must satisfy ``0 < k <= n_cands // 2`` (Weber allows ``k = m/2`` when
        ``m`` is even).
    rng : numpy.random.Generator, optional
        Random number generator.  If omitted, ``numpy.random.default_rng()``
        is used.

    Returns
    -------
    election : ndarray
        A 2D collection of combined approval ballots (``int8``).

    References
    ----------
    .. [1] Weber, Robert J. (1978). "Comparison of Public Choice Systems".
       Cowles Foundation Discussion Papers. Cowles Foundation for Research in
       Economics. No. 498. https://cowles.yale.edu/publications/cfdp/cfdp-498

    """
    utilities = np.asarray(utilities)
    n_voters, n_cands = utilities.shape
    if not 0 < k <= n_cands // 2:
        raise ValueError(
            f'k of {k} not possible for vote-for-or-against-k with '
            f'{n_cands} candidates (require 0 < k <= n_cands // 2)'
        )

    rng = np.random.default_rng(rng)
    ballots = np.zeros((n_voters, n_cands), dtype=np.int8)
    rows = np.arange(n_voters)[:, np.newaxis]

    u = utilities.astype(np.float64, copy=False)
    u_j = u + rng.random(u.shape) * (np.finfo(np.float64).eps * 64)
    top_k = np.argpartition(u_j, -k, axis=1)[:, -k:]
    bot_k = np.argpartition(u_j, k - 1, axis=1)[:, :k]
    ballots[rows, top_k] = 1
    ballots[rows, bot_k] = -1
    return ballots


def vote_for_or_against_k_uniform_types(utilities, k, rng=None):
    """
    Combined-approval ballots from a uniform draw on Weber's finite type list.

    For each voter, independently pick a uniformly random cardinality-``k``
    subset ``S`` (via random ``argpartition`` keys) and then, with probability
    ``1/2``, assign ``+1`` to every candidate in ``S`` or ``-1`` to every
    candidate in ``S``.  The ``utilities`` array only fixes ballot shape and RNG
    seeding; values do not enter the ballot rule.

    This is the literal equal-probability enumeration over ``2 * (m choose k)``
    types.  Under impartial-culture utilities it does **not** reproduce Weber's
    Table 19 Social Utility Efficiency in the same Monte Carlo setup as
    :func:`vote_for_or_against_k`.

    Parameters
    ----------
    utilities : array_like
        Shape ``(n_voters, n_cands)``; values are not used for the ballot rule.
    k : int
        Must satisfy ``0 < k <= n_cands // 2``.
    rng : numpy.random.Generator, optional
        Random number generator.  If omitted, ``numpy.random.default_rng()``
        is used.

    Returns
    -------
    election : ndarray
        A 2D collection of combined approval ballots (``int8``).

    References
    ----------
    .. [1] Weber, Robert J. (1978). "Comparison of Public Choice Systems".
       Cowles Foundation Discussion Papers. Cowles Foundation for Research in
       Economics. No. 498. https://cowles.yale.edu/publications/cfdp/cfdp-498

    """
    utilities = np.asarray(utilities)
    n_voters, n_cands = utilities.shape
    if not 0 < k <= n_cands // 2:
        raise ValueError(
            f'k of {k} not possible for vote-for-or-against-k with '
            f'{n_cands} candidates (require 0 < k <= n_cands // 2)'
        )

    rng = np.random.default_rng(rng)
    ballots = np.zeros((n_voters, n_cands), dtype=np.int8)
    rows = np.arange(n_voters)[:, np.newaxis]
    keys = rng.random((n_voters, n_cands))
    subset = np.argpartition(keys, -k, axis=1)[:, -k:]
    signs = (1 - 2 * rng.integers(2, size=n_voters, dtype=np.int8))[:, np.newaxis]
    ballots[rows, subset] = signs
    return ballots
