import numpy as np
from runoff import runoff
import random
import pytest


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


def test_one_round():
    # 60% majority, tie between others
    election = np.array([[2, 0, 1],
                         [0, 1, 2],
                         [2, 0, 1],
                         [1, 2, 0],
                         [2, 1, 0],
                         [2, 0, 1],
                         [1, 0, 2],
                         [2, 0, 1],
                         [2, 1, 0],
                         [0, 2, 1]])
    assert runoff(election) == 2
    assert runoff(election, tiebreaker='order') == 2
    assert runoff(election, tiebreaker='random') == 2

    # 50% winner, 30%, 20% for others
    # In this case, Candidate 2 picks up an additional vote in the runoff,
    # making it unambiguous.
    election = np.array([[2, 0, 1],
                         [0, 1, 2],
                         [1, 0, 2],
                         [2, 0, 1],
                         [2, 1, 0],
                         [0, 1, 2],
                         [2, 0, 1],
                         [1, 2, 0],
                         [2, 1, 0],
                         [0, 2, 1]])
    assert runoff(election) == 2
    assert runoff(election, tiebreaker='order') == 2
    assert runoff(election, tiebreaker='random') == 2

    # 50%, 30%, 20%
    # This is ambiguous. It would make sense for the 50% candidate to win
    # outright, but technically they don't have a majority, so we have to
    # eliminate another, so there's now a 50/50 split, and then tiebreak
    # between the two, which might pick a different candidate, even though they
    # got fewer first-preference votes.
    election = np.array([[2, 0, 1],
                         [0, 1, 2],
                         [1, 0, 2],
                         [2, 0, 1],
                         [2, 1, 0],
                         [0, 1, 2],
                         [2, 0, 1],
                         [1, 0, 2],
                         [2, 1, 0],
                         [0, 2, 1]])
    assert runoff(election) is None
    assert runoff(election, tiebreaker='order') == 0
    assert collect_random_results(runoff, election) == {0, 2}

    # 50%, 25%, 25%
    # If 0 eliminated by tiebreak, another transfers to each and 2 wins
    # If 1 eliminated by tiebreak, 2 transfer to 0 and a second tiebreak
    # So either 0 or 2 wins.
    election = np.array([[2, 0, 1],
                         [0, 1, 2],
                         [1, 0, 2],
                         [2, 0, 1],
                         [2, 0, 1],
                         [1, 0, 2],
                         [2, 1, 0],
                         [0, 2, 1]])
    assert runoff(election) is None
    assert runoff(election, tiebreaker='order') == 0
    assert collect_random_results(runoff, election) == {0, 2}

    # 50%, 25%, 25%
    # 0 or 1 is eliminated and transfers votes to the other, making it a tie.
    # So any candidate can win.
    election = np.array([[2, 0, 1],
                         [0, 1, 2],
                         [1, 0, 2],
                         [2, 0, 1],
                         [2, 0, 1],
                         [1, 0, 2],
                         [2, 1, 0],
                         [0, 1, 2]])
    assert runoff(election) is None
    assert runoff(election, tiebreaker='order') == 0
    assert collect_random_results(runoff, election) == {0, 1, 2}

    # 50% exact tie
    election = np.array([[2, 0, 1],
                         [1, 0, 2],
                         [1, 2, 0],
                         [2, 1, 0]])
    assert runoff(election) is None
    assert runoff(election, tiebreaker='order') == 1
    assert collect_random_results(runoff, election) == {1, 2}

    # Complete cycle, anyone can win
    election = np.array([[0, 1, 2],
                         [1, 2, 0],
                         [2, 0, 1]])
    assert runoff(election) is None
    assert runoff(election, tiebreaker='order') == 0
    assert collect_random_results(runoff, election) == {0, 1, 2}


@pytest.mark.parametrize("tiebreaker", [None, 'random', 'order'])
def test_two_rounds(tiebreaker):
    # Standard Tennessee example
    # https://en.wikipedia.org/wiki/Template:Tenn_voting_example
    Memphis, Nashville, Chattanooga, Knoxville = 0, 1, 2, 3
    election = [*42*[[Memphis, Nashville, Chattanooga, Knoxville]],
                *26*[[Nashville, Chattanooga, Knoxville, Memphis]],
                *15*[[Chattanooga, Knoxville, Nashville, Memphis]],
                *17*[[Knoxville, Chattanooga, Nashville, Memphis]],
                ]
    assert runoff(election, tiebreaker) == Nashville

    # Example from Ques 9
    # http://www.yorku.ca/bucovets/4380/exercises/exercises_1_a.pdf
    v, w, x, y, z = 0, 1, 2, 3, 4
    election = [*11*[[v, w, x, y, z]],
                *12*[[w, x, y, z, v]],
                *13*[[x, v, w, y, z]],
                *14*[[y, w, v, z, x]],
                *15*[[z, v, x, w, y]],
                ]
    assert runoff(election, tiebreaker) == y

    # Example from https://en.wikipedia.org/wiki/Two-round_system#Example_I
    # (With some of the rankings made up so that they're all complete rankings)
    Ice_cream, Apple_pie, Fruit, Celery = 0, 1, 2, 3
    election = [*10*[[Ice_cream, Apple_pie, Celery,    Fruit]],
                * 3*[[Apple_pie, Ice_cream, Celery,    Fruit]],
                * 3*[[Apple_pie, Fruit,     Celery,    Ice_cream]],
                * 8*[[Fruit,     Apple_pie, Celery,    Ice_cream]],
                * 1*[[Celery,    Fruit,     Apple_pie, Ice_cream]],
                ]
    assert runoff(election, tiebreaker) == Ice_cream


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
