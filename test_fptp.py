from fptp import fptp
import numpy as np
import pytest
import random


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

    assert fptp(election, tiebreaker) == Memphis

    # Example from Ques 9
    # http://www.yorku.ca/bucovets/4380/exercises/exercises_1_a.pdf
    v, w, x, y, z = 0, 1, 2, 3, 4
    election = [*11*[[v, w, x, y, z]],
                *12*[[w, x, y, z, v]],
                *13*[[x, v, w, y, z]],
                *14*[[y, w, v, z, x]],
                *15*[[z, v, x, w, y]],
                ]

    assert fptp(election, tiebreaker) == z

    # 50% plurality but not strictly majority
    election = np.array([[2, 3, 1, 0],
                         [0, 1, 2, 3],
                         [2, 1, 3, 0],
                         [1, 0, 3, 2]])
    assert fptp(election, tiebreaker) == 2

    # 40% plurality, 30% for others
    election = np.array([[2, 0, 1],
                         [0, 1, 2],
                         [2, 0, 1],
                         [1, 2, 0],
                         [0, 1, 2],
                         [2, 0, 1],
                         [1, 0, 2],
                         [2, 0, 1],
                         [1, 0, 2],
                         [0, 2, 1]])
    assert fptp(election, tiebreaker) == 2

    # Example from
    # http://jlmartin.faculty.ku.edu/~jlmartin/courses/math105-F11/Lectures/chapter1-part2.pdf
    A, B, C, D = 0, 1, 2, 3
    election = [*60*[[A, B, C, D]],
                *40*[[B, D, C, A]],
                ]

    assert fptp(election, tiebreaker) == A


def test_ties():
    # Two-way tie between candidates 1 and 2
    election = np.array([[0, 1, 2],
                         [2, 0, 1],
                         [0, 1, 2],
                         [1, 2, 0],
                         [1, 2, 0],
                         [2, 0, 1],
                         [2, 0, 1],
                         [1, 2, 0],
                         ])
    # No tiebreaker:
    assert fptp(election, tiebreaker=None) is None

    # Mode 'order' should always prefer lowest candidate ID
    assert fptp(election, tiebreaker='order') == 1

    # Mode 'random' should choose all tied candidates at random
    assert collect_random_results(fptp, election) == {1, 2}

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
    assert fptp(election, tiebreaker=None) is None

    # Mode 'order' should always prefer lowest candidate ID
    assert fptp(election, tiebreaker='order') == 0

    # Mode 'random' should choose all tied candidates at random
    assert collect_random_results(fptp, election) == {0, 1, 2}


def test_invalid():
    with pytest.raises(ValueError):
        election = [[0, 1]]
        fptp(election, 'dictator')


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
