import random
import numpy as np
import pytest
from hypothesis.strategies import integers, permutations, lists
from hypothesis import given
from coombs import coombs


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
def test_strict_majority(tiebreaker):
    A, B, C = 0, 1, 2
    election = [[A, B, C],
                [B, C, A],
                [A, C, B],
                ]
    assert coombs(election, tiebreaker) == A

    election = [*3*[[A, B, C]],
                *7*[[B, C, A]],
                *2*[[C, B, A]],
                ]
    assert coombs(election, tiebreaker) == B


def test_no_tiebreak_tied_losers():
    A, B, C = 0, 1, 2
    election = [[A, B, C],
                [B, C, A],
                [A, C, B],
                [B, C, A],
                [C, B, A],
                [C, A, B],
                [C, A, B],
                ]
    assert coombs(election) is None


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
    assert coombs(election) == 2
    assert coombs(election, tiebreaker='order') == 2
    assert coombs(election, tiebreaker='random') == 2

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
    assert coombs(election) == 2
    assert coombs(election, tiebreaker='order') == 2
    assert coombs(election, tiebreaker='random') == 2

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
                         [2, 0, 1],
                         [0, 2, 1]])
    assert coombs(election) is None
    assert coombs(election, tiebreaker='order') == 0
    assert collect_random_results(coombs, election) == {0, 2}

    # 50% exact tie
    election = np.array([[2, 1, 0],
                         [1, 2, 0],
                         [1, 2, 0],
                         [2, 1, 0]])
    assert coombs(election) is None
    assert coombs(election, tiebreaker='order') == 1
    assert collect_random_results(coombs, election) == {1, 2}

    # Complete cycle, anyone can win
    election = np.array([[0, 1, 2],
                         [1, 2, 0],
                         [2, 0, 1]])
    assert coombs(election) is None
    assert coombs(election, tiebreaker='order') == 0
    assert collect_random_results(coombs, election) == {0, 1, 2}


@pytest.mark.parametrize("tiebreaker", [None, 'random', 'order'])
def test_examples(tiebreaker):
    # Standard Tennessee example (two-round)
    # https://en.wikipedia.org/wiki/Template:Tenn_voting_example
    Memphis, Nashville, Chattanooga, Knoxville = 0, 1, 2, 3
    election = [*42*[[Memphis, Nashville, Chattanooga, Knoxville]],
                *26*[[Nashville, Chattanooga, Knoxville, Memphis]],
                *15*[[Chattanooga, Knoxville, Nashville, Memphis]],
                *17*[[Knoxville, Chattanooga, Nashville, Memphis]],
                ]
    assert coombs(election, tiebreaker) == Nashville

    # Four-round example from Ques 9
    # http://www.yorku.ca/bucovets/4380/exercises/exercises_1_a.pdf
    v, w, x, y, z = 0, 1, 2, 3, 4
    election = [*11*[[v, w, x, y, z]],
                *12*[[w, x, y, z, v]],
                *13*[[x, v, w, y, z]],
                *14*[[y, w, v, z, x]],
                *15*[[z, v, x, w, y]],
                ]
    assert coombs(election, tiebreaker) == v

    # Two-round example from
    # http://pi.math.cornell.edu/~ismythe/Lec_04_web.pdf#page=16
    A, B, C = 0, 1, 2
    election = [[A, C, B],
                [A, C, B],
                [B, C, A],
                [B, C, A],
                [C, A, B],
                ]
    assert coombs(election, tiebreaker) == C  # "? C wins under Coombs "?

    # Homework example from
    # https://www.slader.com/discussion/question/the-coombs-method-this-method-is-just-like-the-plurality-with-elimination-method-except-that-in-each/#
    A, B, C, D = 0, 1, 2, 3
    election = [*14*[[A, B, C, D]],
                *10*[[C, B, D, A]],
                * 8*[[D, C, B, A]],
                * 4*[[B, D, C, A]],
                * 1*[[C, D, B, A]],
                ]
    assert coombs(election, tiebreaker) == C

    # Example from http://pi.math.cornell.edu/~ismythe/Lec_05_web.pdf#page=2
    A, B, C, D, E = 0, 1, 2, 3, 4
    election = [*4*[[A, C, D, B, E]],
                *3*[[B, E, D, C, A]],
                *2*[[C, B, D, A, E]],
                *1*[[D, B, E, C, A]],
                *1*[[E, D, B, A, C]],
                ]
    assert coombs(election, tiebreaker) == C


def complete_ranked_ballots(min_cands=3, max_cands=256, min_voters=1,
                            max_voters=1000):
    n_cands = integers(min_value=min_cands, max_value=max_cands)
    return n_cands.flatmap(lambda n: lists(permutations(range(n)),
                                           min_size=min_voters,
                                           max_size=max_voters))


@pytest.mark.parametrize("tiebreaker", ['random', 'order'])
@given(election=complete_ranked_ballots(min_cands=1, max_cands=25,
                                        min_voters=1, max_voters=100))
def test_legit_winner_order(election, tiebreaker):
    election = np.asarray(election)
    n_cands = election.shape[1]
    assert coombs(election, tiebreaker) in range(n_cands)


@given(election=complete_ranked_ballots(min_cands=1, max_cands=25,
                                        min_voters=1, max_voters=100))
def test_legit_winner_none(election):
    election = np.asarray(election)
    n_cands = election.shape[1]
    assert coombs(election) in {None} | set(range(n_cands))


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
