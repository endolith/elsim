import random
import numpy as np
from numpy.testing import assert_array_equal
import pytest
from elsim.strategies import approval_optimal, vote_for_k


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
def test_approval_optimal(tiebreaker):
    utilities = np.array([[0.0, 0.4, 1.0],
                          [1.0, 0.5, 1.0],
                          [0.0, 0.2, 0.0],
                          [0.0, 1.0, 0.7],
                          [0.3, 0.4, 0.6],
                          ])
    assert_array_equal(approval_optimal(utilities), [[0, 0, 1],
                                                     [1, 0, 1],
                                                     [0, 1, 0],
                                                     [0, 1, 1],
                                                     [0, 0, 1],
                                                     ])


@pytest.mark.parametrize("tiebreaker", [None, 'random', 'order'])
def test_vote_for_k(tiebreaker):
    utilities = np.array([[0.0, 0.4, 1.0],
                          [1.0, 0.5, 0.9],
                          [0.0, 0.2, 0.1],
                          [0.0, 1.0, 0.7],
                          [0.3, 0.4, 0.6],
                          ])
    assert_array_equal(vote_for_k(utilities, 1), [[0, 0, 1],
                                                  [1, 0, 0],
                                                  [0, 1, 0],
                                                  [0, 1, 0],
                                                  [0, 0, 1],
                                                  ])

    assert_array_equal(vote_for_k(utilities, 'half'), [[0, 0, 1],
                                                       [1, 0, 0],
                                                       [0, 1, 0],
                                                       [0, 1, 0],
                                                       [0, 0, 1],
                                                       ])

    assert_array_equal(vote_for_k(utilities, 2), [[0, 1, 1],
                                                  [1, 0, 1],
                                                  [0, 1, 1],
                                                  [0, 1, 1],
                                                  [0, 1, 1],
                                                  ])


@pytest.mark.parametrize("k", [0, 3, -1])
def test_invalid_k(k):
    with pytest.raises(ValueError):
        election = [[0, 1],
                    [1, 0]]
        vote_for_k(election, k)


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
