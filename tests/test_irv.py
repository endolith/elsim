import random

import numpy as np
import pytest
from hypothesis.strategies import integers, permutations, lists
from hypothesis import given

from elsim.methods import irv


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
    assert irv(election, tiebreaker) == A

    election = [*3*[[A, B, C]],
                *7*[[B, C, A]],
                *2*[[C, B, A]],
                ]
    assert irv(election, tiebreaker) == B


def test_no_tiebreak_tied_losers():
    A, B, C = 0, 1, 2
    election = [[A, B, C],
                [B, C, A],
                [A, C, B],
                [B, C, A],
                [C, B, A],
                [C, A, B],
                [C, B, A],
                ]
    assert irv(election) is None


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
    assert irv(election) == 2
    assert irv(election, tiebreaker='order') == 2
    assert irv(election, tiebreaker='random') == 2

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
    assert irv(election) == 2
    assert irv(election, tiebreaker='order') == 2
    assert irv(election, tiebreaker='random') == 2

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
    assert irv(election) is None
    assert irv(election, tiebreaker='order') == 0
    assert collect_random_results(irv, election) == {0, 2}

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
    assert irv(election) is None
    assert irv(election, tiebreaker='order') == 0
    assert collect_random_results(irv, election) == {0, 2}

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
    assert irv(election) is None
    assert irv(election, tiebreaker='order') == 0
    assert collect_random_results(irv, election) == {0, 1, 2}

    # 50% exact tie
    election = np.array([[2, 0, 1],
                         [1, 0, 2],
                         [1, 2, 0],
                         [2, 1, 0]])
    assert irv(election) is None
    assert irv(election, tiebreaker='order') == 1
    assert collect_random_results(irv, election) == {1, 2}

    # Complete cycle, anyone can win
    election = np.array([[0, 1, 2],
                         [1, 2, 0],
                         [2, 0, 1]])
    assert irv(election) is None
    assert irv(election, tiebreaker='order') == 0
    assert collect_random_results(irv, election) == {0, 1, 2}


@pytest.mark.parametrize("tiebreaker", [None, 'random', 'order'])
def test_examples(tiebreaker):
    # Standard Tennessee example (three round)
    # https://en.wikipedia.org/wiki/Template:Tenn_voting_example
    Memphis, Nashville, Chattanooga, Knoxville = 0, 1, 2, 3
    election = [*42*[[Memphis, Nashville, Chattanooga, Knoxville]],
                *26*[[Nashville, Chattanooga, Knoxville, Memphis]],
                *15*[[Chattanooga, Knoxville, Nashville, Memphis]],
                *17*[[Knoxville, Chattanooga, Nashville, Memphis]],
                ]
    assert irv(election, tiebreaker) == Knoxville

    # Three-round example from Ques 9
    # http://www.yorku.ca/bucovets/4380/exercises/exercises_1_a.pdf
    v, w, x, y, z = 0, 1, 2, 3, 4
    election = [*11*[[v, w, x, y, z]],
                *12*[[w, x, y, z, v]],
                *13*[[x, v, w, y, z]],
                *14*[[y, w, v, z, x]],
                *15*[[z, v, x, w, y]],
                ]
    assert irv(election, tiebreaker) == w

    # Two-round example from
    # https://en.wikipedia.org/wiki/Instant-runoff_voting#Five_voters,_three_candidates
    Bob, Bill, Sue = 0, 1, 2
    election = np.array([[Bob, Bill, Sue],  # a
                         [Sue, Bob, Bill],  # b
                         [Bill, Sue, Bob],  # c
                         [Bob, Bill, Sue],  # d
                         [Sue, Bob, Bill],  # e
                         ])
    assert irv(election, tiebreaker) == Sue

    # Two-round example from
    # https://en.wikipedia.org/wiki/Condorcet_method#Comparison_with_instant_runoff_and_first-past-the-post_(plurality)
    A, B, C = 0, 1, 2
    election = [*499*[[A, B, C]],
                *  3*[[B, C, A]],
                *498*[[C, B, A]],
                ]
    assert irv(election, tiebreaker) == C  # "IRV elects C"

    # Two-round example from
    # http://pi.math.cornell.edu/~ismythe/Lec_04_web.pdf#page=16
    election = [[A, C, B],
                [A, C, B],
                [B, C, A],
                [B, C, A],
                [C, A, B],
                ]
    assert irv(election, tiebreaker) == A  # "A wins under IRV"

    # Examples from http://pi.math.cornell.edu/~ismythe/Lec_05_web.pdf#page=19
    # Two-round
    election = [*6*[[A, B, C]],
                *5*[[C, A, B]],
                *4*[[B, C, A]],
                *2*[[B, A, C]],
                ]
    assert irv(election, tiebreaker) == A  # A wins IRV

    # Two-round
    election = [*6*[[A, B, C]],
                *5*[[C, A, B]],
                *4*[[B, C, A]],
                *2*[[A, B, C]],
                ]
    assert irv(election, tiebreaker) == C  # C wins IRV

    # Four-round example from
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
    assert irv(election) == r


def test_eliminate_no_votes():
    # First, 0 is eliminated for getting no votes.
    # Then 1 and 2 are tied.
    election = [[1, 0, 2],
                [2, 0, 1]]

    # With no tiebreaker, None is returned because of tie between 1 and 2.
    assert irv(election) is None

    # With order tiebreaker, 2 is eliminated, because lower IDs preferred.
    # Then 1 wins unanimously.
    # If 0 had not been eliminated, 0 would win the tie between 0 and 1.
    assert irv(election, 'order') == 1

    # With random tiebreaker, 0 should never win.
    assert collect_random_results(irv, election) == {1, 2}


def complete_ranked_ballots(min_cands=3, max_cands=256, min_voters=1,
                            max_voters=1000):
    n_cands = integers(min_value=min_cands, max_value=max_cands)
    return n_cands.flatmap(lambda n: lists(permutations(range(n)),
                                           min_size=min_voters,
                                           max_size=max_voters))


@pytest.mark.parametrize("tiebreaker", ['random', 'order'])
@given(election=complete_ranked_ballots(min_cands=1, max_cands=25,
                                        min_voters=1, max_voters=100))
def test_legit_winner(election, tiebreaker):
    n_cands = np.shape(election)[1]
    assert irv(election, tiebreaker) in range(n_cands)


@given(election=complete_ranked_ballots(min_cands=1, max_cands=25,
                                        min_voters=1, max_voters=100))
def test_legit_winner_none(election):
    n_cands = np.shape(election)[1]
    assert irv(election) in {None} | set(range(n_cands))


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
