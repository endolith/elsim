import numpy as np
from itertools import combinations
from numba import njit


@njit
def _pair_tallier(pairs, tally):
    """
    Takes a 3D array of pairs and a 2D tallying array and modifies the tallying
    array in-place to tally the pairs
    """
    for i in range(pairs.shape[0]):
        for j in range(pairs.shape[1]):
            pair = pairs[i][j]
            tally[pair[0], pair[1]] += 1


def ranked_election_to_matrix(election):
    """
    Converts a ranked election to a pairwise comparison matrix

    Each entry in the matrix gives the total number of votes obtained by the
    row candidate over the column candidate.

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
    matrix : array_like
        A pairwise comparison matrix of candidate vs candidate defeats.

        For example, a ``3`` in row 2, column 5, means that 3 voters preferred
        Candidate 2 over Candidate 5.  Candidates are not preferred over
        themselves, so the diagonal is all zeros.
    """
    election = np.asarray(election)
    n_cands = election.shape[1]
    sum_matrix = np.zeros((n_cands, n_cands), dtype=int)

    # Look at every pairwise combination of columns in the election at once
    # All 1st-pref candidates vs 2nd-pref, all 1st-pref vs 3rd-pref, etc.
    pairs = election[:, tuple(combinations(range(n_cands), 2))]

    # Make a tallying array and then use numba to tally into it
    sum_matrix = np.zeros((n_cands, n_cands), dtype=np.uint)
    _pair_tallier(pairs, sum_matrix)
    return sum_matrix


def condorcet_from_matrix(matrix):
    """
    Finds the winner of a ranked ballot election using a Condorcet method

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
    """
    if matrix.shape[0] != matrix.shape[1] or len(matrix.shape) != 2:
        raise ValueError('Input must be n by n square sum matrix')

    n_cands = matrix.shape[0]

    # Brute force
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
    Finds the winner of a ranked ballot election using a Condorcet method

    This does not contain any "tiebreakers"; those will be implemented by
    other methods' functions.  It is not a Condorcet completion method.

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
    """
    matrix = ranked_election_to_matrix(election)
    return condorcet_from_matrix(matrix)


if __name__ == "__main__":
    # Run unit tests, in separate process to avoid warnings about cached
    # modules, printing output line by line in realtime
    from subprocess import Popen, PIPE
    with Popen(['pytest',
                '--tb=short',  # shorter traceback format
                '--hypothesis-show-statistics',
                str('test_condorcet.py')], stdout=PIPE, bufsize=1,
               universal_newlines=True) as p:
        for line in p.stdout:
            print(line, end='')
