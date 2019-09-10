import numpy as np
from numpy.testing import assert_array_equal
import pytest
from hypothesis import given
from hypothesis.strategies import integers, lists, permutations
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

    # Table 3.1 from Mackie - Democracy Defended
    A, B, C, D, E = 0, 1, 2, 3, 4
    election = [*4*[[A, E, D, C, B]],
                *3*[[B, C, E, D, A]],
                *2*[[C, D, E, B, A]],
                ]

    assert_array_equal(ranked_election_to_matrix(election),
                       [[0, 4, 4, 4, 4],
                        [5, 0, 3, 3, 3],
                        [5, 6, 0, 5, 5],
                        [5, 6, 4, 0, 2],
                        [5, 6, 4, 7, 0],
                        ])


def test_condorcet():
    # Standard Tennessee example
    # https://en.wikipedia.org/wiki/Template:Tenn_voting_example
    Memphis, Nashville, Chattanooga, Knoxville = 0, 1, 2, 3
    election = [*42*[[Memphis, Nashville, Chattanooga, Knoxville]],
                *26*[[Nashville, Chattanooga, Knoxville, Memphis]],
                *15*[[Chattanooga, Knoxville, Nashville, Memphis]],
                *17*[[Knoxville, Chattanooga, Nashville, Memphis]],
                ]

    assert condorcet(election) == Nashville

    # Example from Ques 9
    # http://www.yorku.ca/bucovets/4380/exercises/exercises_1_a.pdf
    v, w, x, y, z = 0, 1, 2, 3, 4
    election = [*11*[[v, w, x, y, z]],
                *12*[[w, x, y, z, v]],
                *13*[[x, v, w, y, z]],
                *14*[[y, w, v, z, x]],
                *15*[[z, v, x, w, y]],
                ]

    assert condorcet(election) == v

    # Example from Ques 1
    # http://www.yorku.ca/bucovets/4380/exercises/exercises_1_a.pdf
    v, w, x, y, z = 0, 1, 2, 3, 4
    election = [[w, v, x, y, z],
                [v, x, y, z, w],
                [x, z, v, w, y],
                [y, z, v, w, x],
                [z, w, v, y, x],
                ]

    assert condorcet(election) is None

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

    # Table 3.1 from Mackie - Democracy Defended
    A, B, C, D, E = 0, 1, 2, 3, 4
    election = [*4*[[A, E, D, C, B]],
                *3*[[B, C, E, D, A]],
                *2*[[C, D, E, B, A]],
                ]

    assert condorcet(election) == C  # "and C is the Condorcet winner"

    # Example from
    # https://medium.com/@t2ee6ydscv/how-ranked-choice-voting-elects-extremists-fa101b7ffb8e
    r, b, g, o, y = 0, 1, 2, 3, 4
    election = [*31*[[r, b, g, o, y]],
                * 5*[[b, r, g, o, y]],
                * 8*[[b, g, r, o, y]],
                * 1*[[b, g, o, r, y]],
                * 6*[[g, b, o, r, y]],
                * 1*[[g, b, o, y, r]],
                * 6*[[g, o, b, y, r]],
                * 2*[[o, g, b, y, r]],
                * 5*[[o, g, y, b, r]],
                * 7*[[o, y, g, b, r]],
                *28*[[y, o, g, b, r]],
                ]
    assert condorcet(election) == g


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


def test_invalid():
    with pytest.raises(TypeError):
        election = [[0, 1],
                    [1, 0]]
        condorcet(election, 'random')

    with pytest.raises(ValueError):
        condorcet_from_matrix(np.array([[0, 1]]))

    with pytest.raises(ValueError):
        condorcet(np.array([[0],
                            [1]]))


def test_unanimity_condorcet():
    election = [[3, 0, 1, 2], [3, 0, 2, 1], [3, 2, 1, 0]]
    assert condorcet(election) == 3


# No tiebreaker parameter
def test_degenerate_condorcet_case():
    election = [[0]]
    assert condorcet(election) == 0

    election = [[0], [0], [0]]
    assert condorcet(election) == 0


def complete_ranked_ballots(min_cands=3, max_cands=25, min_voters=1,
                            max_voters=100):
    n_cands = integers(min_value=min_cands, max_value=max_cands)
    return n_cands.flatmap(lambda n: lists(permutations(range(n)),
                                           min_size=min_voters,
                                           max_size=max_voters))


@given(election=complete_ranked_ballots(min_cands=1, max_cands=25,
                                        min_voters=1, max_voters=100))
def test_legit_winner(election):
    election = np.asarray(election)
    n_cands = election.shape[1]
    winner = condorcet(election)
    assert isinstance(winner, (int, type(None)))
    assert winner in set(range(n_cands)) | {None}


@given(election=complete_ranked_ballots(min_cands=2, max_cands=25,
                                        min_voters=1, max_voters=100))
def test_ranked_election_to_matrix(election):
    election = np.asarray(election)
    matrix = ranked_election_to_matrix(election)
    assert matrix.shape == (election.shape[1],)*2
    assert matrix.min() == 0
    assert matrix.max() <= len(election)
    assert_array_equal(np.diagonal(matrix), 0)


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
