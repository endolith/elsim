from itertools import combinations
import numpy as np
from elsim.methods._common import _tally_pairs, njit


def ranked_election_to_matrix(election):
    """
    Convert a ranked election to a pairwise comparison matrix.

    Each entry in the matrix gives the total number of votes obtained by the
    row candidate over the column candidate.[1]_

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
    matrix : ndarray
        A pairwise comparison matrix of candidate vs candidate defeats.

        For example, a ``3`` in row 2, column 5, means that 3 voters preferred
        Candidate 2 over Candidate 5.  Candidates are not preferred over
        themselves, so the diagonal is all zeros.

    References
    ----------
    .. [1] Wikipedia, “Condorcet method: Pairwise counting and matrices”,
       https://en.wikipedia.org/wiki/Condorcet_method#Pairwise_counting_and_matrices

    Examples
    --------
    Label some candidates:

    >>> A, B, C = 0, 1, 2

    Specify the ballots for the 5 voters:

    >>> election = [[A, C, B],
    ...             [A, C, B],
    ...             [B, C, A],
    ...             [B, C, A],
    ...             [C, A, B],
    ...             ]

    Convert to a matrix:

    >>> ranked_election_to_matrix(election)
    array([[0, 3, 2],
           [2, 0, 2],
           [3, 3, 0]], dtype=uint32)

    This shows that Candidate
    A (row 0) is preferred over B (column 1) by 3 voters.
    C (row 2) is preferred over A (column 0) by 3 voters.
    C (row 2) is preferred over B (column 1) by 3 voters.
    """
    election = np.asarray(election)

    # Cache results since this is slow and called repeatedly with same input
    self = ranked_election_to_matrix
    try:
        if (self._cache_in == election.tobytes() and
                self._cache_shape == election.shape):
            return self._cache_out
    except AttributeError:
        self._cache_in = None
        self._cache_shape = None
        self._cache_out = None

    n_cands = election.shape[1]
    sum_matrix = np.zeros((n_cands, n_cands), dtype=int)

    # Look at every pairwise combination of columns in the election at once
    # All 1st-pref candidates vs 2nd-pref, all 1st-pref vs 3rd-pref, etc.
    pairs = election[:, tuple(combinations(range(n_cands), 2))]

    # Make a tallying array and then tally all pairs into it
    sum_matrix = np.zeros((n_cands, n_cands), dtype=np.uint)
    _tally_pairs(pairs, sum_matrix)

    self._cache_in = election.tobytes()
    self._cache_shape = election.shape
    self._cache_out = sum_matrix

    return sum_matrix


@njit(cache=True, nogil=True)
def condorcet_from_matrix(matrix):
    """
    Find the winner of a ranked ballot election using a Condorcet method.

    This does not contain any "tiebreakers"; those will be implemented by
    other methods' functions.  It is not a Condorcet completion method.

    Parameters
    ----------
    matrix : array_like
        A pairwise comparison matrix of candidate vs candidate defeats.

    Returns
    -------
    winner : int
        The ID number of the winner, or ``None`` for a Condorcet cycle / tie.

    Examples
    --------
    Label some candidates:

    >>> A, B, C = 0, 1, 2

    Specify the pairwise comparison matrix for the election:

    >>> matrix = np.array([[0, 3, 2],
    ...                    [2, 0, 2],
    ...                    [3, 3, 0]])

    Candidate
    A (row 0) is preferred over B (column 1) by 3 voters.
    C (row 2) is preferred over A (column 0) by 3 voters.
    C (row 2) is preferred over B (column 1) by 3 voters.
    C is thus the Condorcet winner:

    >>> condorcet_from_matrix(matrix)
    2
    """
    # TODO: np.asarray would be nice for accepting array_like, but breaks numba
    if matrix.shape[0] != matrix.shape[1] or len(matrix.shape) != 2:
        raise ValueError('Input must be n by n square sum matrix')

    n_cands = matrix.shape[0]

    # Vectorized version is actually slower:
    # matrix = matrix.astype(np.uint)
    # n_cands = len(matrix)
    # wins = (matrix > matrix.T).sum(axis=1)
    # winner = (wins == n_cands-1).nonzero()[0]
    # if len(winner) == 1:
    #     return int(winner[0])
    # else:
    #     return None

    # Brute force numba version is 4× faster (in part because it halts early)
    for runner in range(n_cands):
        wins = 0
        for opponent in range(n_cands):
            if runner == opponent:
                continue
            else:
                if matrix[runner, opponent] > matrix[opponent, runner]:
                    wins += 1
        if wins == n_cands - 1:
            return runner
    return None


def condorcet(election):
    """
    Find the winner of a ranked ballot election using a Condorcet method.

    This does not contain any "tiebreakers"; those will be implemented by
    other methods' functions.  It is not a Condorcet completion method.[1]_

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
        The ID number of the winner, or ``None`` for a Condorcet cycle / tie.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Condorcet_method

    Examples
    --------
    Label some candidates:

    >>> A, B, C = 0, 1, 2

    Specify the ballots for the 5 voters:

    >>> election = [[A, C, B],
    ...             [A, C, B],
    ...             [B, C, A],
    ...             [B, C, A],
    ...             [C, A, B],
    ...             ]

    Candidate
    A is preferred over B by 3 voters.
    C is preferred over A by 3 voters.
    C is preferred over B by 3 voters.
    C is thus the Condorcet winner:

    >>> condorcet(election)
    2
    """
    election = np.asarray(election)

    # Handle special case of 1 candidate
    if election.shape[1] == 1:
        if all(election == 0):
            return 0
        else:
            raise ValueError('Election is not a set of complete ranking '
                             'ballots')

    matrix = ranked_election_to_matrix(election)
    return condorcet_from_matrix(matrix)
