import numpy as np
from numpy.testing import assert_array_equal
from condorcet import (condorcet, condorcet_from_matrix,
                       ranked_election_to_matrix)


def test_ranked_election_to_matrix_basic():
    # Test examples from
    # https://en.wikipedia.org/wiki/Condorcet_method#Pairwise_counting_and_matrices
    r"""
    B > C > A > D

    which should produce:

    R\O	A	B	C	D
    A 	—	0	0	1
    B 	1	—	1	1
    C 	1	0	—	1
    D 	0	0	0	—
    """
    A, B, C, D = 0, 1, 2, 3
    election = np.asarray([[B, C, A, D]])
    assert_array_equal(ranked_election_to_matrix(election),
                       [[0, 0, 0, 1],
                        [1, 0, 1, 1],
                        [1, 0, 0, 1],
                        [0, 0, 0, 0],
                        ])

    r"""
    B > C > A > D
    D > A > C > B
    A > C > B > D

    which should produce:

    R\O	A	B	C	D
    A 	— 	2	2	2
    B 	1	—	1	2
    C 	1	2	—	2
    D 	1	1	1	—
    """
    election = np.asarray([[B, C, A, D],
                           [D, A, C, B],
                           [A, C, B, D],
                           ])
    assert_array_equal(ranked_election_to_matrix(election),
                       [[0, 2, 2, 2],
                        [1, 0, 1, 2],
                        [1, 2, 0, 2],
                        [1, 1, 1, 0],
                        ])

    # Table 9 from
    # https://www.researchgate.net/publication/265892455_On_the_Relevance_of_Theoretical_Results_to_Voting_System_Choice
    election = [*5*[[D, B, C, A]],
                *4*[[B, C, A, D]],
                *3*[[A, D, C, B]],
                *3*[[A, D, B, C]],
                *4*[[C, A, B, D]],
                ]
    assert_array_equal(ranked_election_to_matrix(election),
                       [[ 0, 10,  6, 14],
                        [ 9,  0, 12,  8],
                        [13,  7,  0,  8],
                        [ 5, 11, 11,  0],
                        ])


def test_condorcet():
    # Example from
    # https://en.wikipedia.org/wiki/Condorcet_method#Pairwise_counting_and_matrices
    election = np.array([[1, 2, 0, 3],
                         [3, 0, 2, 1],
                         [0, 2, 1, 3],
                         ])
    assert condorcet(election) == 0

    # Example from https://electowiki.org/wiki/Condorcet_Criterion
    election = np.concatenate((np.tile([0, 1, 2], (499, 1)),
                               np.tile([2, 1, 0], (498, 1)),
                               np.tile([1, 2, 0], (3, 1)),
                               ))
    assert condorcet(election) == 1

    # Example from
    # https://www3.nd.edu/~apilking/Math10170/Information/Lectures/Lecture_3.Head%20To%20Head%20Comparisons.pdf
    Colley, Henry, Taylor = 0, 1, 2
    election = np.array([[Colley,  Henry, Taylor],
                         [ Henry, Colley, Taylor],
                         [ Henry, Colley, Taylor],
                         [Taylor, Colley,  Henry],
                         [Taylor,  Henry, Colley],
                         ])
    assert condorcet(election) == 1

    # Example from https://www.whydomath.org/node/voting/impossible.html
    election = np.array([[0, 2, 3, 1],
                         [1, 2, 3, 0],
                         [3, 0, 2, 1],
                         [0, 1, 3, 2],
                         [3, 0, 2, 1],
                         ])
    assert condorcet(election) == 3


def test_condorcet_from_matrix():
    # Examples from Young 1988 Condorcet's Theory of Voting
    # http://www.cs.cmu.edu/~arielpro/15896s15/docs/paper4a.pdf
    # Table 1
    matrix = np.array([[0,  8,  6],
                       [5,  0, 11],
                       [7,  2,  0]])
    assert condorcet_from_matrix(matrix) is None

    # Table 2
    matrix = np.array([[ 0, 12, 15, 17],
                       [13,  0, 16, 11],
                       [10,  9,  0, 18],
                       [ 8, 14,  7,  0]])
    assert condorcet_from_matrix(matrix) is None

    # Table 4
    matrix = np.array([[ 0, 23, 29],
                       [37,  0, 29],
                       [31, 31,  0]])
    assert condorcet_from_matrix(matrix) == 2

    # Examples from https://rangevoting.org/EMorg/CondorcetEx.htm
    matrix = np.array([[ 0, 63, 89, 57],
                       [87,  0, 78, 73],
                       [69, 72,  0, 74],
                       [67, 51, 52,  0]])
    assert condorcet_from_matrix(matrix) == 1

    matrix = np.array([[ 0, 40, 22, 13],
                       [37,  0, 50, 50],
                       [30, 35,  0, 25],
                       [20, 60, 20,  0]])
    assert condorcet_from_matrix(matrix) is None

    # Example from https://bolson.org/voting/VRRexplaination.pdf
    # Is actually a cycle, but they illustrate a tiebreaker method, not named
    matrix = np.array([[ 0, 16, 14, 16],
                       [15,  0, 17, 20],
                       [17, 14,  0, 16],
                       [15, 11, 15,  0]])
    assert condorcet_from_matrix(matrix) is None

    # Example from https://electowiki.org/wiki/Schulze_method#Example_1
    # These are all Condorcet failures, though
    matrix = np.array([[ 0, 20, 26, 30, 22],
                       [25,  0, 16, 33, 18],
                       [19, 29,  0, 17, 24],
                       [15, 12, 28,  0, 14],
                       [23, 27, 21, 31,  0]])
    assert condorcet_from_matrix(matrix) is None

    # Debian election from https://www.debian.org/vote/2003/vote_0001
    matrix = np.array([[  0,  34,  66,  38, 228],
                       [428,   0, 238, 224, 449],
                       [385, 221,   0, 226, 405],
                       [397, 228, 237,   0, 424],
                       [202,  29,  65,  39,   0]])
    assert condorcet_from_matrix(matrix) == 3


if __name__ == "__main__":
    # Run unit tests, in separate process to avoid warnings about cached
    # modules, printing output line by line in realtime
    from subprocess import Popen, PIPE
    with Popen(['pytest',
                '--tb=short',  # shorter traceback format
                '--hypothesis-show-statistics',
                str(__file__)], stdout=PIPE, bufsize=1,
               universal_newlines=True) as p:
        for line in p.stdout:
            print(line, end='')
