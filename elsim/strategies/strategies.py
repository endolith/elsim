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
                     [0.1, 0.8, 1.0],
                     [0.0, 0.5, 0.5],
                     ]

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


def vote_for_or_against_k(utilities, k):
    """
    Convert utilities to combined approval voting ballots, with `k` marks.

    Given a set of utilities for each voter-candidate pair, if the candidates'
    average utilities for a voter are closer to 0, they are modeled as voting
    for the top `k` candidates.  If the average is closer to 1, they are
    modeled as voting against their least favorite `k` candidates.  With a
    random society (IC) model, these two strategies will be equally likely.[1]_

    Parameters
    ----------
    utilities : array_like
        A 2D collection of utilities.

        Rows represent voters, and columns represent candidate IDs.
        Higher utility numbers mean greater approval of that candidate by that
        voter.
    k : int
        The number of candidates approved or disapproved of by each voter.

    Returns
    -------
    election : ndarray
        A 2D collection of approval ballots.

        Rows represent voters, and columns represent candidate IDs.
        A cell contains 1 if that voter approves of that candidate, a -1 if
        the voter disapproves of that candidate, or a 0 if they are neutral.

    References
    ----------
    .. [1] Weber, Robert J. (1978). "Comparison of Public Choice Systems".
       Cowles Foundation Discussion Papers. Cowles Foundation for Research in
       Economics. No. 498. https://cowles.yale.edu/publications/cfdp/cfdp-498

    Examples
    --------
    Voter 0 loves Candidates A (index 0) and B (index 1), but hates C, D, and
    E (2, 3, 4).
    Voter 1 dislikes A and E, likes B and D, and loves C.
    Voter 2 hates A, dislikes C and E, and is lukewarm about B and D.

    >>> utilities = [[1.0, 1.0, 0.0, 0.0, 0.0],
                     [0.1, 0.8, 1.0, 0.9, 0.2],
                     [0.0, 0.5, 0.1, 0.5, 0.1],
                     ]

    If `k` = 2, voters are required to vote for or against 2 candidates.

    Since Voter 0 likes 2 candidates and hates 3, they choose to vote *for*
    their two favorites.

    Since Voter 1 likes 3 candidates and dislikes 2, they choose to vote
    *against* their two least favorites.

    Since Voter 2 dislikes 3 candidates, they choose to vote *for* the two
    candidates they are lukewarm about:

    >>> vote_for_or_against_k(utilities, 2)
    array([[+1, +1, 0,  0,  0],
           [-1,  0, 0,  0, -1],
           [ 0, +1, 0, +1,  0]], dtype=int8)

    If `k` were 3, however, Voters 0 and 2 would choose to vote *against* their
    three disliked candidates instead, while Voter 1 would switch to voting
    *for* their 3 liked candidates:

    >>> vote_for_or_against_k(utilities, 3)
    array([[ 0,  0, -1, -1, -1],
           [ 0, +1, +1, +1,  0],
           [-1,  0, -1,  0, -1]], dtype=int8)

    """
    utilities = np.asarray(utilities)
    n_cands = utilities.shape[1]
    if k <= 0 or k > n_cands - 1:
        raise ValueError(f'k of {k} not possible with {n_cands} candidates')

    # TODO: Does Weber model them as setting their threshold to 0.5, or the
    # midpoint of their range?  For general use it should be midpoint of range

    # Measure min and max for each voter, and then check if it's above THEIR OWN average utility
    # For each voter, calculate average utility and midpoint utility
    # TODO: Make a utility funciont for this
    def mid(utilities):
        return (utilities.max(axis=1) + utilities.min(axis=1)) / 2      # or np.mean of the two values?
    voter_mid = mid(utilities)

    # Vfoa1 doesn't quite match.  Is this what Weber did?
    voter_mid[:] = 0.5

    voter_mean = utilities.mean(axis=1)

    # Or maybe he literally means that each voter chooses to approve or
    # disapprove at random?


    # # TODO: Simplify all this
    # # Actually, Weber specifies k < m/2
    # if k < n_cands / 2:
    #     # If average utility < mid, then do approve of top k
    #     # If average utility > mid, then do disapprove of bottom k
    #     approving_voters = np.where(voter_mean < voter_mid)[0]
    #     disapproving_voters = np.where(voter_mean >= voter_mid)[0]
    # else:
    #     approving_voters = np.where(voter_mean >= voter_mid)[0]
    #     disapproving_voters = np.where(voter_mean < voter_mid)[0]


    # "a voter is equally likely to be of either type."
    approving_voters = np.arange(len(utilities)//2)
    disapproving_voters = np.arange(len(utilities)//2, len(utilities))

    # Create blank ballots
    approvals = np.zeros(utilities.shape, np.int8)

    # slice utilities to get just approving voters?  then find top k of that slice
    approving_utilities = utilities[approving_voters]

    # slice utilities to get just disapproving voters, then find bottom k of that slice?
    disapproving_utilities = utilities[disapproving_voters]

    # https://stackoverflow.com/a/23734295/125507
    top_k = np.argpartition(approving_utilities, -k, axis=1)[:, -k:]
    bot_k = np.argpartition(disapproving_utilities, k, axis=1)[:, :k]
    # function this, used in multiple

    """
    img
    Out[55]:
    array([[  0., 255.,   0.,   0.,   0.],
           [  0.,   0., 255.,   0.,   0.],
           [  0.,   0.,   0., 255.,   0.],
           [  0.,   0.,   0.,   0., 255.],
           [  0.,   0.,   0.,   0.,   0.]])

    img[((0, 1, 2, 3), (1, 2, 3, 4))]


    so if I have
        approving_voters = [0, 2]

    and
        top_k = [[1, 0],
                 [1, 3]]

    Then I want to feed these indices to the assignment:
        ((0, 0, 2, 2), (1, 0, 1, 3))

    and for
        disapproving_voters = [1]

    and
        bot_k = [[0, 4]]

    I want
        ((1, 1), (0, 4))


    Likewise, for
        approving_voters = [1]
    and
        top_k = [[1, 3, 2]]
    I want
        ((1, 1, 1), (1, 3, 2))

    and for
        disapproving_voters = [0, 2]
    and
        bot_k = [[3, 2, 4],
                 [0, 2, 4]]
    I want
        (0, 0, 0, 2, 2, 2), (3, 2, 4, 0, 2, 4)

    Why can I not figure this out

    0    3 2 4
    2    0 2 4

    chain([0 for x in [3, 2, 4]], [2 for x in [0, 2, 4]])

    try this:
        x[row_indices,:][:,col_indices]

    utilities[disapproving_voters, :][:, bot_k]

    nope.

    IS THIS WHAT I WANT?
        utilities[disapproving_voters][:,bot_k]
        no.

        utilities[disapproving_voters][bot_k]
        no.

        utilities[disapproving_voters,:][bot_k]
        no.

        utilities[disapproving_voters][bot_k,:]
        no.

        utilities[:,disapproving_voters][bot_k]
        no.

        utilities[:,disapproving_voters][bot_k,:]
        no.

    WAIT

    disapproving voters isinside a tuple.  added a [0] to get it out

    utilities[np.ix_(approving_voters, top_k)]
    nope.


    Just do it explicitly

    rows = []
    cols = []
    for r in range(len(bot_k)):
        for c in range(len(bot_k[0])):
            print(disapproving_voters[r], bot_k[r][c])
            rows.append(disapproving_voters[r])
            cols.append(bot_k[r][c])

    that works!

    so as a comprehension:

    [disapproving_voters[r] for r in range(len(bot_k)) for c in range(len(bot_k[0]))]
    [bot_k[r][c] for r in range(len(bot_k)) for c in range(len(bot_k[0]))]

    found through stupefying trial and error

    [disapproving_voters[r] for r in range(len(bot_k)) for c in range(len(bot_k[0]))]
    [bot_k[r][c] for r in range(len(bot_k)) for c in range(len(bot_k[0]))]

    [(r, c) for r in range(len(bot_k)) for c in range(len(bot_k[0]))]

    ok whatever it works.  deal with simplifying it some other day

    """

    # TODO there must be a simpler way to do this, ask on stackoverflow
    rows = []
    cols = []
    for r in range(len(top_k)):
        for c in range(len(top_k[0])):
            rows.append(approving_voters[r])
            cols.append(top_k[r][c])

    # assign +1 for approving voters
    approvals[rows, cols] = 1

    rows = []
    cols = []
    for r in range(len(bot_k)):
        for c in range(len(bot_k[0])):
            rows.append(disapproving_voters[r])
            cols.append(bot_k[r][c])

    # and -1 for disapproving voters
    approvals[rows, cols] = -1

    #TODO THIS IS UGLY CLEAN IT UP AND COMMENT IT

    return approvals
