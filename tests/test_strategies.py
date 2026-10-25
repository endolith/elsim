import numpy as np
from numpy.testing import assert_array_equal
import pytest
from hypothesis import given
from hypothesis.strategies import integers, tuples, floats
from hypothesis.extra.numpy import arrays
from elsim.strategies import (approval_optimal, vote_for_k,
                              honest_normed_scores,
                              vote_for_or_against_k)


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


def test_approval_optimal():
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


def test_honest_normed_scores():
    utilities = np.array([[0.0, 0.4, 1.0],
                          [1.0, 0.5, 1.0],
                          [0.0, 0.2, 0.0],
                          [0.0, 1.0, 0.7],
                          [0.3, 0.4, 0.6],
                          ])
    assert_array_equal(honest_normed_scores(utilities, 7), [[0, 3, 7],
                                                            [7, 0, 7],
                                                            [0, 7, 0],
                                                            [0, 7, 5],
                                                            [0, 2, 7],
                                                            ])


def test_vote_for_k():
    utilities = np.array([[0.0, 0.4, 1.0],
                          [1.0, 0.5, 0.9],
                          [0.0, 0.2, 0.1],
                          [0.0, 1.0, 0.7],
                          [0.3, 0.4, 0.6],
                          ])

    a = [[0, 0, 1],
         [1, 0, 0],
         [0, 1, 0],
         [0, 1, 0],
         [0, 0, 1]]
    assert_array_equal(vote_for_k(utilities, 1), a)
    assert_array_equal(vote_for_k(utilities, -2), a)
    assert_array_equal(vote_for_k(utilities, 'half'), a)

    b = [[0, 1, 1],
         [1, 0, 1],
         [0, 1, 1],
         [0, 1, 1],
         [0, 1, 1]]
    assert_array_equal(vote_for_k(utilities, 2), b)
    assert_array_equal(vote_for_k(utilities, -1), b)



def test_vote_for_or_against_k():
    utilities = np.array([[0.0, 0.4, 1.0],
                          [1.0, 0.5, 0.9],
                          [0.0, 0.2, 0.3],
                          [0.0, 1.0, 0.7],
                          [0.3, 0.4, 0.6],
                          ])
    # k = 1, three candidates: approve the top 1, or the top 2 (disapprove the
    # bottom 1) if the middle candidate is above the voter's average utility.
    assert_array_equal(vote_for_or_against_k(utilities, 1), [[0, 0, 1],
                                                             [1, 0, 1],
                                                             [0, 1, 1],
                                                             [0, 1, 1],
                                                             [0, 0, 1],
                                                             ])

    utilities = np.array([[1.0, 1.0, 0.0, 0.0, 0.0],
                          [0.1, 0.8, 1.0, 0.9, 0.2],
                          [0.0, 0.5, 0.1, 0.5, 0.1],
                          ])
    assert_array_equal(vote_for_or_against_k(utilities, 2), [[1, 1, 0, 0, 0],
                                                             [0, 1, 1, 1, 0],
                                                             [0, 1, 0, 1, 0],
                                                             ])

    # Each voter approves of either k or m-k candidates, never anything else
    rng = np.random.default_rng(0)
    for m in range(2, 12):
        for k in range(1, m//2 + 1):
            ballots = vote_for_or_against_k(rng.random((50, m)), k)
            assert set(np.unique(ballots)) <= {0, 1}
            assert set(np.unique(ballots.sum(axis=1))) <= {k, m - k}


@pytest.mark.parametrize("method", [vote_for_k, vote_for_or_against_k])
@pytest.mark.parametrize("k", [0, 3, -3, -4, 4])
def test_invalid_k(k, method):
    with pytest.raises(ValueError):
        election = [[0.0, 0.5, 1.0],
                    [1.0, 0.0, 0.1]]
        method(election, k)


def utilities(min_cands=2, max_cands=25, min_voters=1, max_voters=100):
    """
    Strategy to generate utilities arrays
    """
    n_cands = integers(min_value=min_cands, max_value=max_cands)
    n_voters = integers(min_value=min_voters, max_value=max_voters)
    return arrays(float, tuples(n_voters, n_cands), elements=floats(0, 1))


@given(utilities=utilities())
def test_approval_optimal_properties(utilities):
    election = approval_optimal(utilities)
    assert election.shape == utilities.shape
    assert set(election.flat) <= {0, 1}


@given(utilities=utilities())
def test_vote_for_k_properties(utilities):
    election = vote_for_k(utilities, 1)
    assert election.shape == utilities.shape
    assert 1 in set(election.flat)
    assert set(election.flat) <= {0, 1}


@given(utilities=utilities(), max_score=integers(1, 100))
def test_honest_normed_scores_properties(utilities, max_score):
    election = honest_normed_scores(utilities, max_score)
    assert election.shape == utilities.shape

    # Normalized should contain both 0 and max_score for all voters. However,
    # Hypothesis will find degenerate cases with indifferent voters. (TODO)
    assert_array_equal(election.min(axis=1), 0)
    assert election.min() == 0
    assert election.min() <= max_score

    # Output should be integers
    assert_array_equal(election % 1, 0)


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
