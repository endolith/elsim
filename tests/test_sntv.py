import random

import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import integers, lists, permutations

from elsim.methods import sntv


def collect_random_results(method, election):
    """
    Run multiple elections with tiebreaker='random' and collect the set of all
    winners.
    """
    random.seed(47)  # Deterministic test
    winners = set()
    for trial in range(10):
        winner = method(election, tiebreaker='random')
        assert isinstance(winner, set)
        winners.update(winner)
    return winners


@pytest.mark.parametrize("tiebreaker", [None, 'random', 'order'])
def test_basic(tiebreaker):
    # https://en.wikipedia.org/wiki/Single_non-transferable_vote#Example
    A, B, C, D, E = 0, 1, 2, 3, 4
    election = [* 819*[A],
                *1804*[B],
                *1996*[C],
                *1999*[D],
                *2718*[E],
                ]
    assert sntv(election, 3, tiebreaker) == {C, D, E}

    # Example from Ques 9
    # http://www.yorku.ca/bucovets/4380/exercises/exercises_1_a.pdf
    v, w, x, y, z = 0, 1, 2, 3, 4
    election = [*11*[[v, w, x, y, z]],
                *12*[[w, x, y, z, v]],
                *13*[[x, v, w, y, z]],
                *14*[[y, w, v, z, x]],
                *15*[[z, v, x, w, y]],
                ]

    assert sntv(election, tiebreaker=tiebreaker) == {z}
    assert sntv(election, 1, tiebreaker) == {z}

    # 50% plurality but not strictly majority
    election = np.array([[2, 3, 1, 0],
                         [0, 1, 2, 3],
                         [2, 1, 3, 0],
                         [1, 0, 3, 2]])
    assert sntv(election, tiebreaker=tiebreaker) == {2}
    assert sntv(election, 1, tiebreaker) == {2}

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
    assert sntv(election, tiebreaker=tiebreaker) == {2}
    assert sntv(election, 1, tiebreaker) == {2}

    # Example from
    # http://jlmartin.faculty.ku.edu/~jlmartin/courses/math105-F11/Lectures/chapter1-part2.pdf
    A, B, C, D = 0, 1, 2, 3
    election = [*60*[[A, B, C, D]],
                *40*[[B, D, C, A]],
                ]

    assert sntv(election, tiebreaker=tiebreaker) == {A}
    assert sntv(election, 1, tiebreaker) == {A}

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
    assert sntv(election, tiebreaker=tiebreaker) == {r}
    assert sntv(election, 1, tiebreaker) == {r}


def test_ties():
    # Two-way tie between candidates 1 and 2 (ranked ballots)
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
    assert sntv(election, tiebreaker=None) is None

    # Mode 'order' should always prefer lowest candidate ID
    assert sntv(election, tiebreaker='order') == {1}

    # Mode 'random' should choose all tied candidates at random
    assert collect_random_results(sntv, election) == {1, 2}

    # Three-way tie between 0, 1, and 2 (single-mark ballots)
    election = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

    # No tiebreaker:
    assert sntv(election, tiebreaker=None) is None

    # Mode 'order' should always prefer lowest candidate ID
    assert sntv(election, tiebreaker='order') == {0}

    # Mode 'random' should choose all tied candidates at random
    assert collect_random_results(sntv, election) == {0, 1, 2}

    # Tie between candidates that exceed n
    election = np.array([0, 0, 0, 1, 1, 2, 2, 3])

    # No tiebreaker:
    assert sntv(election, n=2, tiebreaker=None) is None

    # Mode 'order' should always prefer lowest candidate ID
    assert sntv(election, n=2, tiebreaker='order') == {0, 1}

    # Mode 'random' should always choose 0, and tied candidates at random
    assert 0 in sntv(election, n=2, tiebreaker='order')
    assert 3 not in sntv(election, n=2, tiebreaker='order')
    # TODO: There needs to be a collect_random_results for multi-winner
    # assert collect_random_results(sntv, election) == {0, 1, 2}

    # Same but with different IDs
    election = np.array([3, 3, 3, 1, 1, 0, 0, 2])

    # No tiebreaker:
    assert sntv(election, n=2, tiebreaker=None) is None

    # Mode 'order' should always prefer lowest candidate ID
    assert sntv(election, n=2, tiebreaker='order') == {3, 0}

    # Mode 'random' should always choose 0, and tied candidates at random
    assert 3 in sntv(election, n=2, tiebreaker='order')
    assert 2 not in sntv(election, n=2, tiebreaker='order')

    # Same but with different IDs, order
    election = np.array([3, 1, 2, 0, 2, 2, 1, 3])

    # No tiebreaker:
    assert sntv(election, n=2, tiebreaker=None) is None

    # Mode 'order' should always prefer lowest candidate ID
    assert sntv(election, n=2, tiebreaker='order') == {2, 1}

    # Mode 'random' should always choose 0, and tied candidates at random
    assert 2 in sntv(election, n=2, tiebreaker='order')
    assert 0 not in sntv(election, n=2, tiebreaker='order')


@pytest.mark.parametrize("tiebreaker", [None, 'random', 'order'])
def test_1d(tiebreaker):
    # Standard Tennessee example
    # https://en.wikipedia.org/wiki/Template:Tenn_voting_example
    Memphis, Nashville, Chattanooga, Knoxville = 0, 1, 2, 3
    election = [*42*[Memphis], *26*[Nashville], *15*[Chattanooga],
                *17*[Knoxville]]
    assert sntv(election, tiebreaker=tiebreaker) == {Memphis}
    assert sntv(election, 1, tiebreaker) == {Memphis}

    # Example from Ques 9
    # http://www.yorku.ca/bucovets/4380/exercises/exercises_1_a.pdf
    v, w, x, y, z = 0, 1, 2, 3, 4
    election = [*11*[v], *12*[w], *13*[x], *14*[y], *15*[z]]
    assert sntv(election, tiebreaker=tiebreaker) == {z}
    assert sntv(election, 1, tiebreaker) == {z}

    # 50% plurality but not strictly majority
    election = np.array([2, 0, 2, 1])
    assert sntv(election, tiebreaker=tiebreaker) == {2}
    assert sntv(election, 1, tiebreaker) == {2}

    # 40% plurality, 30% for others
    election = np.array([2, 0, 2, 1, 0, 2, 1, 2, 1, 0])
    assert sntv(election, tiebreaker=tiebreaker) == {2}
    assert sntv(election, 1, tiebreaker) == {2}

    # Example from
    # http://jlmartin.faculty.ku.edu/~jlmartin/courses/math105-F11/Lectures/chapter1-part2.pdf
    A, B = 0, 1
    election = [*60*[A], *40*[B]]
    assert sntv(election, tiebreaker=tiebreaker) == {A}
    assert sntv(election, 1, tiebreaker) == {A}


def test_invalid():
    with pytest.raises(ValueError):
        sntv(np.array([[[0, 1]]]))


def complete_ranked_ballots(min_cands=3, max_cands=25, min_voters=1,
                            max_voters=100):
    """
    Strategy to generate complete ranked ballot arrays, like those produced by
    elections.impartial_culture()
    """
    n_cands = integers(min_value=min_cands, max_value=max_cands)
    return n_cands.flatmap(lambda n: lists(permutations(range(n)),
                                           min_size=min_voters,
                                           max_size=max_voters))


@pytest.mark.parametrize("tiebreaker", ['random', 'order'])
@given(election=complete_ranked_ballots(min_cands=1, max_cands=25,
                                        min_voters=1, max_voters=100),
       n=integers(min_value=1, max_value=25))
def test_legit_winner(election, tiebreaker, n):
    n_cands = np.shape(election)[1]
    winners = sntv(election, n, tiebreaker)
    assert isinstance(winners, set)
    assert 0 < len(winners) <= n_cands
    assert winners <= set(range(n_cands))


@given(election=complete_ranked_ballots(min_cands=1, max_cands=25,
                                        min_voters=1, max_voters=100),
       n=integers(min_value=1, max_value=25))
def test_legit_winner_none(election, n):
    n_cands = np.shape(election)[1]
    winners = sntv(election, n)
    assert isinstance(winners, (set, type(None)))
    if winners is not None:
        assert 0 < len(winners) <= n_cands
        assert winners <= set(range(n_cands))


@pytest.mark.parametrize("tiebreaker", ['random', 'order'])
@given(election=lists(integers(min_value=1, max_value=25),
                      min_size=1, max_size=100),
       n=integers(min_value=1, max_value=25))
def test_legit_winner_single_mark(election, tiebreaker, n):
    n_cands = 26  # TODO: Vary this?
    winners = sntv(election, n, tiebreaker)
    assert isinstance(winners, set)
    assert 0 < len(winners) <= n_cands
    assert winners <= set(range(n_cands))


@given(election=lists(integers(min_value=1, max_value=25),
                      min_size=1, max_size=100),
       n=integers(min_value=1, max_value=25))
def test_legit_winner_none_single_mark(election, n):
    n_cands = 26  # TODO: Vary this?
    winners = sntv(election, n)
    assert isinstance(winners, (set, type(None)))
    if winners is not None:
        assert 0 < len(winners) <= n_cands
        assert winners <= set(range(n_cands))


@pytest.mark.parametrize("tiebreaker", ['random', 'order'])
def test_small_n(tiebreaker):
    # number of voters less than n
    election = [5, 7]
    assert sntv(election, n=2, tiebreaker=tiebreaker) == {5, 7}
    assert sntv(election, n=7, tiebreaker=tiebreaker) == {5, 7}

    # number of voted candidates less than n
    election = [5, 5, 5, 5, 6, 6, 6, 7, 7, 7]
    assert sntv(election, n=3, tiebreaker=tiebreaker) == {5, 6, 7}
    assert sntv(election, n=9, tiebreaker=tiebreaker) == {5, 6, 7}


@given(election=complete_ranked_ballots(min_cands=1, max_cands=25,
                                        min_voters=1, max_voters=100))
def test_defaults(election):
    assert sntv(election) == sntv(election, 1) == sntv(election, 1, None)


if __name__ == "__main__":
    # Run unit tests, in separate process to avoid warnings about cached
    # modules, printing output line by line in realtime
    from subprocess import PIPE, Popen
    with Popen(['pytest',
                '--tb=short',  # shorter traceback format
                '--hypothesis-show-statistics',
                str(__file__)], stdout=PIPE, bufsize=1,
               universal_newlines=True) as p:
        for line in p.stdout:
            print(line, end='')
