import numpy as np
import pytest
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats, integers, tuples
from numpy.testing import assert_array_equal

from elsim.strategies import (approval_optimal, honest_normed_scores,
                              honest_rankings, vote_for_k)


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

    uniform = np.full((3, 4), 0.25)
    assert_array_equal(
        approval_optimal(uniform), np.zeros((3, 4), dtype=np.uint8)
    )


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

    indifferent = np.ones((4, 5))
    assert_array_equal(
        honest_normed_scores(indifferent, max_score=9),
        np.zeros((4, 5), dtype=np.uint8),
    )


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

    # Strict total order per row: each voter approves exactly k candidates.
    strict = (
        np.linspace(0.0, 1.0, 5, dtype=np.float64)
        + np.arange(4, dtype=np.float64)[:, np.newaxis] * 1e-4
    )
    assert_array_equal(vote_for_k(strict, 1).sum(axis=1), np.ones(4))
    assert_array_equal(vote_for_k(strict, 4).sum(axis=1), np.full(4, 4))


@pytest.mark.parametrize("k", [0, 3, -3, -4, 4])
def test_invalid_k(k):
    with pytest.raises(ValueError):
        election = [[0.0, 0.5, 1.0],
                    [1.0, 0.0, 0.1]]
        vote_for_k(election, k)


def test_honest_rankings_complete_indifference_ordering():
    utilities = np.array([[0.0, 0.0, 0.0]])
    assert_array_equal(honest_rankings(utilities), [[2, 1, 0]])


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
    means = utilities.mean(axis=1, keepdims=True)
    assert np.all((election == 0) | (utilities > means))


@given(utilities=utilities())
def test_vote_for_k_properties(utilities):
    election = vote_for_k(utilities, 1)
    assert election.shape == utilities.shape
    assert 1 in set(election.flat)
    assert set(election.flat) <= {0, 1}


@given(utilities=utilities(), max_score=integers(1, 100))
def test_honest_normed_scores_properties(utilities, max_score):
    utilities = np.asarray(utilities, dtype=np.float64)
    utilities = utilities + np.linspace(
        0.0, 1e-7, utilities.shape[1], dtype=np.float64
    )
    election = honest_normed_scores(utilities, max_score)
    assert election.shape == utilities.shape

    assert_array_equal(election.min(axis=1), 0)
    assert_array_equal(election.max(axis=1), max_score)
    assert election.min() <= max_score

    # Output should be integers
    assert_array_equal(election % 1, 0)


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
