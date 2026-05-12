import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats, integers, tuples
from numpy.testing import assert_array_equal

from elsim.strategies import (approval_optimal, honest_normed_scores,
                              vote_for_k, vote_for_or_against_k)


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


@pytest.mark.parametrize("k", [0, 3, -3, -4, 4])
def test_invalid_k(k):
    with pytest.raises(ValueError):
        election = [[0.0, 0.5, 1.0],
                    [1.0, 0.0, 0.1]]
        vote_for_k(election, k)


def test_vote_for_or_against_k_shape():
    rng = np.random.default_rng(0)
    utilities = rng.random((50, 7))
    k = 3
    b = vote_for_or_against_k(utilities, k, rng=rng)
    assert b.shape == utilities.shape
    assert b.dtype == np.int8
    assert set(np.unique(b)) <= {-1, 0, 1}
    assert_array_equal((b == 1).sum(axis=1), np.full(50, k))
    assert_array_equal((b == -1).sum(axis=1), np.full(50, k))
    assert_array_equal((b == 0).sum(axis=1), np.full(50, 7 - 2 * k))


def test_vote_for_or_against_k_uniform_types_shape():
    rng = np.random.default_rng(1)
    utilities = rng.random((50, 7))
    k = 3
    b = vote_for_or_against_k(utilities, k, rng=rng, strategy='uniform_types')
    assert b.shape == utilities.shape
    assert b.dtype == np.int8
    assert set(np.unique(b)) <= {-1, 0, 1}
    assert_array_equal(np.abs(b).sum(axis=1), np.full(50, k))
    assert_array_equal((b == 0).sum(axis=1), np.full(50, 7 - k))
    pos = (b == 1).sum(axis=1) == k
    neg = (b == -1).sum(axis=1) == k
    assert_array_equal(pos | neg, np.ones(50, dtype=bool))


def test_vote_for_or_against_k_invalid_strategy():
    utilities = np.random.default_rng(2).random((4, 8))
    with pytest.raises(ValueError):
        vote_for_or_against_k(utilities, 2, strategy='bogus')


@pytest.mark.parametrize("k", [0, 4])
def test_vote_for_or_against_k_invalid_k(k):
    utilities = np.random.default_rng(1).random((4, 7))
    with pytest.raises(ValueError):
        vote_for_or_against_k(utilities, k)


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


@given(utilities=utilities(min_cands=4, max_cands=20))
def test_vote_for_or_against_k_extremal_properties(utilities):
    m = utilities.shape[1]
    k = m // 2
    assume(k >= 1)
    rng = np.random.default_rng(0)
    b = vote_for_or_against_k(utilities, k, rng=rng)
    assert b.shape == utilities.shape
    assert set(b.flat) <= {-1, 0, 1}
    assert_array_equal((b == 1).sum(axis=1), np.full(utilities.shape[0], k))
    assert_array_equal((b == -1).sum(axis=1), np.full(utilities.shape[0], k))


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
    from subprocess import PIPE, Popen
    with Popen(['pytest',
                '--tb=short',  # shorter traceback format
                '--hypothesis-show-statistics',
                str(__file__)], stdout=PIPE, bufsize=1,
               universal_newlines=True) as p:
        for line in p.stdout:
            print(line, end='')
