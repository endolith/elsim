import random
import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import integers, tuples, floats
from hypothesis.extra.numpy import arrays
from elsim.methods import utility_winner


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
    utilities = np.array([[0.0, 0.5, 1.0],
                          [1.0, 0.5, 1.0],
                          [0.0, 0.2, 0.0],
                          [0.0, 1.0, 0.7],
                          [0.3, 0.4, 0.5],
                          ])
    assert utility_winner(utilities, tiebreaker) == 2


def test_ties():
    # Ties are very unlikely because of float equality

    # Two-way tie between candidates 1 and 2
    utilities = np.array([[0.0, 0.5, 1.0],
                          [1.0, 0.5, 0.0],
                          [0.0, 0.5, 0.0],
                          [0.0, 1.0, 1.0],
                          [0.2, 0.0, 0.5],
                          ])
    # No tiebreaker:
    assert utility_winner(utilities, tiebreaker=None) is None

    # Mode 'order' should always prefer lowest candidate ID
    assert utility_winner(utilities, tiebreaker='order') == 1

    # Mode 'random' should choose all tied candidates at random
    assert collect_random_results(utility_winner, utilities) == {1, 2}

    # Three-way tie between 0, 1, and 2
    utilities = np.array([[0.0, 0.5, 1.0],
                          [1.0, 0.5, 0.0],
                          [0.0, 1.0, 0.5],
                          [1.0, 0.0, 0.5],
                          [0.5, 0.5, 0.5],
                          ])

    # No tiebreaker:
    assert utility_winner(utilities, tiebreaker=None) is None

    # Mode 'order' should always prefer lowest candidate ID
    assert utility_winner(utilities, tiebreaker='order') == 0

    # Mode 'random' should choose all tied candidates at random
    assert collect_random_results(utility_winner, utilities) == {0, 1, 2}


def random_utilities(min_cands=1, max_cands=25, min_voters=1, max_voters=100):
    """
    Strategy to generate utility arrays like those produced by
    elections.random_utilities()
    """
    n_cands = integers(min_value=min_cands, max_value=max_cands)
    n_voters = integers(min_value=min_voters, max_value=max_voters)
    return arrays(float, tuples(n_voters, n_cands), elements=floats(0, 1))


@pytest.mark.parametrize("tiebreaker", ['random', 'order'])
@given(election=random_utilities(min_cands=1, max_cands=25,
                                 min_voters=1, max_voters=100))
def test_legit_winner(election, tiebreaker):
    election = np.asarray(election)
    n_cands = election.shape[1]
    winner = utility_winner(election, tiebreaker)
    assert isinstance(winner, int)
    assert winner in range(n_cands)


@given(election=random_utilities(min_cands=1, max_cands=25,
                                 min_voters=1, max_voters=100))
def test_legit_winner_no_tiebreaker(election):
    election = np.asarray(election)
    n_cands = election.shape[1]
    winner = utility_winner(election)
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
