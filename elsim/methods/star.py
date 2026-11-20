import numpy as np

from elsim.methods._common import (_all_indices, _get_tiebreak, _no_tiebreak,
                                   _order_tiebreak_keep, _random_tiebreak)
from elsim.methods.condorcet import condorcet_from_matrix

_tiebreak_map = {'order': _order_tiebreak_keep,
                 'random': _random_tiebreak,
                 None: _no_tiebreak}


def matrix_from_scores(election):
    """
    Convert a scored ballot election to a pairwise comparison matrix.

    Parameters
    ----------
    election : array_like
        A 2D collection of score ballots.

        Rows represent voters, and columns represent candidate IDs.
        A cell contains a high score if that voter approves of that candidate,
        or low score if they disapprove.

    Returns
    -------
    matrix : ndarray
        A pairwise comparison matrix of candidate vs candidate defeats.

        For example, a ``3`` in row 2, column 5, means that 3 voters preferred
        Candidate 2 over Candidate 5.  Candidates are not preferred over
        themselves, so the diagonal is all zeros.

    Examples
    --------
    Voter 0 loves Candidates A (index 0) and B (index 1), but hates C (2).
    Voter 1 dislikes A, likes B, and loves C.
    Voter 2 hates A, is lukewarm about B and likes C.

    >>> election = [[5, 5, 0],
    ...             [0, 4, 5],
    ...             [0, 3, 4]]

    So, candidate B is preferred over A by 2 voters, while one is indifferent
    between them.  C is preferred over A by 2 voters, while A is preferred
    over C by 1 voter.  C is preferred over B by 2 voters, while B is preferred
    over C by 1 voter.

    >>> matrix_from_scores(election)
    array([[0, 0, 1],
           [2, 0, 1],
           [2, 2, 0]])

    """
    # Add extra dimensions so that election is broadcast against itself,
    # producing every combination of candidate pairs. Count how often each
    # beats the other.
    gt = np.expand_dims(election, 2) > np.expand_dims(election, 1)

    # Sum across all voters
    return gt.sum(axis=0)


def _pairwise_compare(election, a, b):
    """
    Find more-preferred candidate between `a` and `b` in `election`.

    Parameters
    ----------
    election : array_like
        A 2D collection of score ballots.
    a, b : int
        Indices of candidates.

    Returns
    -------
    winner : {a, b, None}
        Index of candidate who beats the other, or None if there is a tie.

    """
    a_beats_b = (election[:, a] > election[:, b]).sum()
    b_beats_a = (election[:, b] > election[:, a]).sum()
    if a_beats_b > b_beats_a:
        return a
    elif b_beats_a > a_beats_b:
        return b
    else:
        return None


def _scorewise_compare(tallies, a, b):
    """
    Find higher-scored candidate between `a` and `b` in `election`.

    Parameters
    ----------
    tallies : array_like
        A 1D collection of tallies of scores
    a, b : int
        Indices of candidates.

    Returns
    -------
    winner : {a, b, None}
        Index of candidate who beats the other, or None if there is a tie.

    """
    if tallies[a] > tallies[b]:
        return a
    elif tallies[b] > tallies[a]:
        return b
    else:
        return None


def _all_condorcet_from_matrix(matrix):
    """
    Find all winners of a ranked ballot election using a Condorcet method.

    This does not contain any "tiebreakers"; it returns all tied Condorcet
    winners.

    This is the "matches won" step of STAR's recommended Condorcet
    tiebreaker.  The full protocol also compares preference votes and win
    margin before a random tiebreak; see
    https://www.starvoting.org/condorcet_tiebreaker.

    Parameters
    ----------
    matrix : array_like
        A pairwise comparison matrix of candidate vs candidate defeats.

    Returns
    -------
    winners : ndarray
        Array of ID numbers of winners.

    Examples
    --------
    Specify the pairwise comparison matrix for the election:

    >>> matrix = np.array([[0, 7, 4, 4],
    ...                    [3, 0, 6, 4],
    ...                    [6, 4, 0, 4],
    ...                    [3, 3, 3, 0]])

    Candidate
    A (row 0) is preferred over B (column 1) and D (column 3).
    B (row 1) is preferred over C and D.
    C is preferred over A and D.
    So A, B, and C form a top cycle.

    >>> _all_condorcet_from_matrix(matrix)
    array([0, 1, 2]...)

    TODO: Is this actually correct? Handle cycles with unequal wins correctly.
    """
    if matrix.shape[0] != matrix.shape[1] or len(matrix.shape) != 2:
        raise ValueError('Input must be n by n square sum matrix')

    matrix = matrix.astype(np.uint)
    wins = (matrix > matrix.T).sum(axis=1)
    winners = (wins == wins.max()).nonzero()[0]
    return winners


def _star_condorcet_candidate(election, candidates, tiebreak,
                               copeland_only):
    """
    Choose one candidate from a set tied during the scoring round.

    A Condorcet comparison is made between the tied candidates.  If it has a
    winner, that candidate is returned.  Otherwise the `tiebreak` callable
    chooses among the tied candidates.

    Parameters
    ----------
    election : array_like
        A 2D collection of score ballots, as accepted by `star`.
    candidates : sequence of int
        Candidate IDs tied in the scoring round.
    tiebreak : callable
        Tiebreaker from ``_tiebreak_map``.
    copeland_only : bool
        If True, restrict an unbroken Condorcet comparison to the candidates
        with the most pairwise wins.  If False, every candidate in `candidates`
        is available to the tiebreaker.

    Returns
    -------
    winner : {int, None}
        The ID number of the chosen candidate, or ``None`` if the tie could not
        be broken.
    """
    matrix = matrix_from_scores(election[:, candidates])
    winner = condorcet_from_matrix(matrix)
    if winner is not None:
        return candidates[winner]

    if copeland_only:
        # Matches-won step; preference votes and win margin are intentionally
        # omitted here.
        candidates = [candidates[i]
                      for i in _all_condorcet_from_matrix(matrix)]
    result = tiebreak(candidates, 1)
    # TODO: don't convert back and forth?
    return result[0]


def _star_scoring_round(election, tallies, tiebreak):
    """
    Find the two STAR runoff finalists, breaking scoring-round ties.

    Parameters
    ----------
    election : array_like
        A 2D collection of score ballots, as accepted by `star`.
    tallies : array_like
        Total score for each candidate.
    tiebreak : callable
        Tiebreaker from ``_tiebreak_map``.

    Returns
    -------
    first, second : {int, None}
        The runoff finalists.  When `second` is ``None``, `first` is either the
        already-decided winner or ``None`` if the scoring tie could not be
        broken.
    decided : bool
        True when the scoring round already determined the winner.
    """
    highest = max(tallies)
    first_set = _all_indices(tallies, highest)

    if len(first_set) == 2:
        # 2 tied for highest score: Both advance to runoff.
        return first_set[0], first_set[1], False

    if len(first_set) > 2:
        # 3 or more tied for highest score: Tiebreak using Condorcet.
        # "2. Ties in the Scoring round should be broken in favor of the
        # candidate who was preferred head-to-head by more voters."
        winner = _star_condorcet_candidate(election, first_set, tiebreak,
                                           copeland_only=True)
        return winner, None, True

    # One candidate has the highest score.
    first = first_set[0]
    # Find the set of candidates who have the second-highest score
    second_score = np.sort(tallies)[-2]
    second_set = _all_indices(tallies, second_score)
    if len(second_set) == 1:
        # One candidate has second-highest score. They go to runoff.
        return first, second_set[0], False

    # Two or more candidates have second-highest score.
    # "2. Ties in the Scoring Round should be broken in favor of the
    # candidate who was preferred head-to-head by more voters."
    second = _star_condorcet_candidate(election, second_set, tiebreak,
                                       copeland_only=False)
    return first, second, False


def _star_runoff(election, tallies, first, second, tiebreak):
    """
    Run the STAR automatic runoff between two finalists.

    Parameters
    ----------
    election : array_like
        A 2D collection of score ballots, as accepted by `star`.
    tallies : array_like
        Total score for each candidate.
    first, second : int
        Candidate IDs of the runoff finalists.
    tiebreak : callable
        Tiebreaker from ``_tiebreak_map``.

    Returns
    -------
    winner : {int, None}
        The ID number of the winner, or ``None`` for an unbroken tie.
    """
    winner = _pairwise_compare(election, first, second)
    if winner is not None:
        return winner

    # "1. Ties in the Runoff Round should be broken in favor of the candidate
    # who was scored higher if possible."
    winner = _scorewise_compare(tallies, first, second)
    if winner is not None:
        return winner

    # '3. Ties which can be broken as above are known as simple ties and
    # should be broken as above. Ties which can not be broken as above are
    # considered "True Ties."'
    # "In the event that a true tie arises you can opt to resolve it with a
    # random tiebreaker"
    return tiebreak([first, second], 1)[0]


def star(election, tiebreaker=None):
    """
    Find the winner of an election using STAR voting.

    The more-preferred of the two highest-scoring candidates wins. [1]_ [2]_

    Parameters
    ----------
    election : array_like
        A 2D collection of score ballots.

        Rows represent voters, and columns represent candidate IDs.
        A cell contains a high score if that voter approves of that candidate,
        or low score if they disapprove.

    tiebreaker : {'random', 'order', None}, optional
        If there is a tie, and `tiebreaker` is ``'random'``, a random finalist
        is returned.
        If 'order', the lowest-ID tied candidate is returned.
        By default, ``None`` is returned for ties.

    Returns
    -------
    winner : {int, None}
        The ID number of the winner, or ``None`` for an unbroken tie.

    Notes
    -----
    If there is a tie during the scoring round (three or more candidates tied
    for highest score, or two or more tied for second highest), it is broken
    using a Condorcet method between the tied candidates. [3]_

    If there is a tie during the automatic runoff (neither candidate is
    preferred more than the other) then it is broken in favor of the
    higher-scoring candidate. [3]_

    If there is still a tie in either round, it is broken according to the
    `tiebreaker` parameter.

    References
    ----------
    .. [1] :doi:`10.1007/s10602-022-09389-3`
    .. [2] :wikipedia:`STAR voting`
    .. [3] https://www.starvoting.us/ties

    Examples
    --------
    Voter 0 loves Candidates A (index 0) and B (index 1), but hates C (2).
    Voter 1 dislikes A, likes B, and loves C.
    Voter 2 hates A, is lukewarm about B and likes C.

    >>> election = [[5, 5, 0],
    ...             [0, 4, 5],
    ...             [0, 3, 4]]

    Candidates B and C (1 and 2) get the highest scores (12 and 9).  Of these,
    C is preferred on more ballots, and wins the election:

    >>> star(election)
    2

    """
    election = np.asarray(election)

    if election.min() < 0:
        raise ValueError('Scores cannot be negative')

    if election.shape[1] == 1:
        # Only 1 candidate: that candidate wins.
        return 0

    # Break any True Ties using specified method
    tiebreak = _get_tiebreak(tiebreaker, _tiebreak_map)

    # Tally all scores
    tallies = election.sum(axis=0)

    first, second, decided = _star_scoring_round(election, tallies, tiebreak)
    if decided:
        return first
    if second is None:
        return None

    return _star_runoff(election, tallies, first, second, tiebreak)
