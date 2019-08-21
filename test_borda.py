import numpy as np
import random
from borda import borda
import pytest
from hypothesis import given
from hypothesis.strategies import integers, lists, permutations


def collect_random_results(method, election):
    """
    Run multiple elections with tiebreaker='random' and collect the set of all
    winners.
    """
    random.seed(47)  # Deterministic test
    winners = set()
    for trial in range(10):
        winner = method(election, tiebreaker='random')
        assert isinstance(winner, int)
        winners.add(winner)
    return winners


@pytest.mark.parametrize("tiebreaker", [None, 'random', 'order'])
def test_basic(tiebreaker):
    # Standard Tennessee example
    # https://en.wikipedia.org/wiki/Template:Tenn_voting_example
    Memphis, Nashville, Chattanooga, Knoxville = 0, 1, 2, 3
    election = [*42*[[Memphis, Nashville, Chattanooga, Knoxville]],
                *26*[[Nashville, Chattanooga, Knoxville, Memphis]],
                *15*[[Chattanooga, Knoxville, Nashville, Memphis]],
                *17*[[Knoxville, Chattanooga, Nashville, Memphis]],
                ]

    assert borda(election, tiebreaker) == Nashville

    # Example from Ques 9
    # http://www.yorku.ca/bucovets/4380/exercises/exercises_1_a.pdf
    v, w, x, y, z = 0, 1, 2, 3, 4
    election = [*11*[[v, w, x, y, z]],
                *12*[[w, x, y, z, v]],
                *13*[[x, v, w, y, z]],
                *14*[[y, w, v, z, x]],
                *15*[[z, v, x, w, y]],
                ]

    assert borda(election, tiebreaker) == w

    # Manually calculated correct answer
    election = [[0, 1, 4, 3, 2],
                [4, 2, 3, 1, 0],
                [4, 2, 3, 1, 0],
                [3, 2, 1, 4, 0],
                [2, 0, 3, 1, 4],
                [3, 2, 1, 4, 0],
                ]

    assert borda(election, tiebreaker) == 2

    # Example from
    # https://www3.nd.edu/~apilking/math10170/information/Lectures/Lecture-2.Borda%20Method.pdf
    K, H, R = 0, 1, 2
    election = [*2*[[K, H, R]],
                *3*[[H, R, K]],
                *2*[[H, K, R]],
                *3*[[R, H, K]],
                ]

    assert borda(election, tiebreaker) == H

    # Example from
    # http://jlmartin.faculty.ku.edu/~jlmartin/courses/math105-F11/Lectures/chapter1-part2.pdf
    A, B, C, D = 0, 1, 2, 3
    election = [*14*[[A, B, C, D]],
                *10*[[C, B, D, A]],
                * 8*[[D, C, B, A]],
                * 4*[[B, D, C, A]],
                * 1*[[C, D, B, A]],
                ]

    assert borda(election, tiebreaker) == B

    election = [*60*[[A, B, C, D]],
                *40*[[B, D, C, A]],
                ]

    assert borda(election, tiebreaker) == B

    # Table 3.1 from Mackie - Democracy Defended
    A, B, C, D, E = 0, 1, 2, 3, 4
    election = [*4*[[A, E, D, C, B]],
                *3*[[B, C, E, D, A]],
                *2*[[C, D, E, B, A]],
                ]

    assert borda(election, tiebreaker) == E  # "to E the Borda winner"


def test_ties():
    # Two-way tie between candidates 1 and 2
    election = np.array([[0, 1, 2],
                         [0, 2, 1],
                         [1, 2, 0],
                         [1, 2, 0],
                         [1, 2, 0],
                         [2, 1, 0],
                         [2, 1, 0],
                         [2, 1, 0],
                         ])
    # No tiebreaker:
    assert borda(election, tiebreaker=None) is None

    # Mode 'order' should always prefer lowest candidate ID
    assert borda(election, tiebreaker='order') == 1

    # Mode 'random' should choose all tied candidates at random
    assert collect_random_results(borda, election) == {1, 2}

    # Three-way tie between 0, 1, and 2
    election = np.array([[0, 1, 2],
                         [0, 1, 2],
                         [0, 1, 2],
                         [1, 2, 0],
                         [1, 2, 0],
                         [1, 2, 0],
                         [2, 0, 1],
                         [2, 0, 1],
                         [2, 0, 1],
                         ])

    # No tiebreaker:
    assert borda(election, tiebreaker=None) is None

    # Mode 'order' should always prefer lowest candidate ID
    assert borda(election, tiebreaker='order') == 0

    # Mode 'random' should choose all tied candidates at random
    assert collect_random_results(borda, election) == {0, 1, 2}


def complete_ranked_ballots(min_cands=3, max_cands=25, min_voters=1,
                            max_voters=100):
    n_cands = integers(min_value=min_cands, max_value=max_cands)
    return n_cands.flatmap(lambda n: lists(permutations(range(n)),
                                           min_size=min_voters,
                                           max_size=max_voters))


@pytest.mark.parametrize("tiebreaker", ['random', 'order'])
@given(election=complete_ranked_ballots(min_cands=1, max_cands=25,
                                        min_voters=1, max_voters=100))
def test_legit_winner(election, tiebreaker):
    election = np.asarray(election)
    n_cands = election.shape[1]
    winner = borda(election, tiebreaker)
    assert isinstance(winner, int)
    assert winner in range(n_cands)


@given(election=complete_ranked_ballots(min_cands=1, max_cands=25,
                                        min_voters=1, max_voters=100))
def test_legit_winner_none(election):
    election = np.asarray(election)
    n_cands = election.shape[1]
    winner = borda(election)
    assert isinstance(winner, (int, type(None)))
    assert winner in set(range(n_cands)) | {None}


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
