import random
import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import integers, tuples
from hypothesis.extra.numpy import arrays
from elsim.methods import score


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


# Score should pass same tests as Approval, and some of the same as STAR (TODO)
@pytest.mark.parametrize("method", [score])
@pytest.mark.parametrize("tiebreaker", [None, 'random', 'order'])
def test_basic(tiebreaker, method):
    # Standard Tennessee example
    # https://electowiki.org/wiki/Score_voting#Example
    #                 Memphis Nashville Chattanooga Knoxville
    election = [*42*[[10,     4,        2,          0]],
                *26*[[0,     10,        4,          2]],
                *15*[[0,      6,       10,          6]],
                *17*[[0,      5,        7,         10]],
                ]

    assert method(election, tiebreaker) == 1  # Nashville

    # Example from http://www-users.math.umn.edu/~rogness/math1001/approval/
    #                 Gore McCain Bush
    election = [*20*[[1,   0,     0]],
                *35*[[1,   1,     0]],
                *25*[[0,   1,     1]],
                *20*[[0,   0,     1]],
                ]

    assert method(election, tiebreaker) == 1  # McCain

    # Example from
    # https://bluecc.instructure.com/courses/4190/files/244440/
    #                 A  B  C  D
    election = [*32*[[1, 0, 0, 1]],
                *31*[[0, 1, 0, 0]],
                *30*[[1, 0, 1, 0]],
                *27*[[0, 1, 0, 1]],
                *30*[[0, 1, 1, 1]],
                *30*[[1, 0, 1, 0]],
                ]

    assert method(election, tiebreaker) == 0  # A


@pytest.mark.parametrize("method", [score])
def test_ties(method):
    # Two-way tie between candidates 1 and 2
    election = np.array([[0, 3, 2],
                         [3, 0, 1],
                         [0, 1, 3],
                         [0, 2, 0],
                         [0, 3, 3],
                         ])
    # No tiebreaker:
    assert method(election, tiebreaker=None) is None

    # Mode 'order' should always prefer lowest candidate ID
    assert method(election, tiebreaker='order') == 1

    # Mode 'random' should choose all tied candidates at random
    assert collect_random_results(method, election) == {1, 2}

    # Three-way tie between 0, 1, and 2
    election = np.array([[0, 1, 1],
                         [0, 1, 2],
                         [0, 2, 1],
                         [2, 4, 0],
                         [4, 1, 0],
                         [1, 3, 0],
                         [1, 0, 3],
                         [3, 0, 1],
                         [1, 0, 4],
                         ])

    # No tiebreaker:
    assert method(election, tiebreaker=None) is None

    # Mode 'order' should always prefer lowest candidate ID
    assert method(election, tiebreaker='order') == 0

    # Mode 'random' should choose all tied candidates at random
    assert collect_random_results(method, election) == {0, 1, 2}


def score_ballots(min_cands=1, max_cands=25, min_voters=1, max_voters=100):
    """
    Strategy to generate score voting ballot elections
    """
    n_cands = integers(min_value=min_cands, max_value=max_cands)
    n_voters = integers(min_value=min_voters, max_value=max_voters)
    return arrays(np.uint, tuples(n_voters, n_cands), elements=integers(0, 99))


@pytest.mark.parametrize("method", [score])
@pytest.mark.parametrize("tiebreaker", ['random', 'order'])
@given(election=score_ballots(min_cands=1, max_cands=25,
                              min_voters=1, max_voters=100))
def test_legit_winner(election, tiebreaker, method):
    election = np.asarray(election)
    n_cands = election.shape[1]
    winner = method(election, tiebreaker)
    assert isinstance(winner, int)
    assert winner in range(n_cands)


@pytest.mark.parametrize("method", [score])
@given(election=score_ballots(min_cands=1, max_cands=25,
                              min_voters=1, max_voters=100))
def test_legit_winner_none(election, method):
    election = np.asarray(election)
    n_cands = election.shape[1]
    winner = method(election)
    assert isinstance(winner, (int, type(None)))
    assert winner in set(range(n_cands)) | {None}


@pytest.mark.parametrize("method", [score])
def test_invalid(method):
    with pytest.raises(ValueError):
        method([[-2, -2, -2],
                [0, +1, -1]])


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
