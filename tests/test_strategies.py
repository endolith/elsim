import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
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


def test_approval_optimal_ignores_candidates_at_mean_utility():
    """
    Optimal approval approves only candidates strictly above the voter's mean
    utility.  A candidate exactly at the mean is not approved.
    (Strict > follows Weber’s greater-than-average definition.)
    """
    utilities = np.array([[0.0, 1.0, 2.0]])
    assert_array_equal(approval_optimal(utilities), [[0, 0, 1]])


def test_honest_rankings_candidate_limit():
    """
    uint8 ranking ballots support at most 255 candidates.  255 is valid and
    256 raises; off-by-one mistakes in the limit would accept 256 or reject
    255.
    """
    rankings = honest_rankings(np.zeros((1, 255)))
    assert rankings.shape == (1, 255)
    assert rankings.max() == 254

    with pytest.raises(ValueError):
        honest_rankings(np.zeros((1, 256)))


def test_honest_normed_scores_default_max_score():
    """The default maximum score is 5, as documented for 0-5 score ballots."""
    utilities = np.array([[0.0, 1.0]])
    assert_array_equal(honest_normed_scores(utilities), [[0, 5]])


@pytest.mark.parametrize(("n_cands", "n_approved"), [(4, 2), (5, 2), (6, 3),
                                                     (7, 3)])
def test_vote_for_k_half_approves_half_the_candidates(n_cands, n_approved):
    """
    The 'half' shortcut uses n_cands // 2, for both even and odd candidate
    counts.
    """
    utilities = np.arange(n_cands, dtype=float)[np.newaxis, :]
    election = vote_for_k(utilities, 'half')
    assert election.sum() == n_approved
    assert_array_equal(election[0],
                       [0]*(n_cands - n_approved) + [1]*n_approved)


def test_vote_for_k_large_k_approves_exactly_top_k():
    """
    k can exceed half the candidates; the ballot must still approve exactly the
    top k utilities, not an arbitrary argpartition slice.
    """
    utilities = np.array([[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]])
    assert_array_equal(vote_for_k(utilities, 4)[0], [0, 0, 1, 1, 1, 1])


@pytest.mark.parametrize("k", [0, 3, -3, -4, 4])
def test_invalid_k(k):
    election = [[0.0, 0.5, 1.0],
                [1.0, 0.0, 0.1]]
    with pytest.raises(ValueError):
        vote_for_k(election, k)


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


def integer_utilities(min_cands=2, max_cands=8, min_voters=1,
                      max_voters=20):
    """Strategy for small integer-valued utilities, to hit exact means."""
    n_cands = integers(min_value=min_cands, max_value=max_cands)
    n_voters = integers(min_value=min_voters, max_value=max_voters)
    return arrays(np.int64, tuples(n_voters, n_cands),
                  elements=integers(0, 5)).map(lambda a: a.astype(float))


@given(data=st.data())
def test_vote_for_k_approves_the_top_k_utilities(data):
    """
    vote_for_k approves exactly k candidates per voter, and every approved
    utility is at least every rejected utility.  Negative k is equivalent to
    counting from the other end.
    """
    utils = data.draw(utilities(min_cands=2, max_cands=8, min_voters=1,
                                max_voters=20), label="utilities")
    n_cands = utils.shape[1]
    k = data.draw(integers(1, n_cands - 1), label="k")

    election = vote_for_k(utils, k)
    assert_array_equal(election, vote_for_k(utils, k - n_cands))
    assert_array_equal(election.sum(axis=1), k)

    for row, ballot in zip(utils, election, strict=True):
        approved = row[ballot == 1]
        rejected = row[ballot == 0]
        assert approved.min() >= rejected.max()

    assert_array_equal(vote_for_k(utils, 'half').sum(axis=1),
                       n_cands // 2)


@given(utilities=integer_utilities())
def test_approval_optimal_approves_strictly_above_mean(utilities):
    """
    Optimal approval approves exactly the candidates whose utility is strictly
    greater than the voter's mean utility; utility equal to the mean is not
    approved.
    """
    means = utilities.mean(axis=1, keepdims=True)
    assert_array_equal(approval_optimal(utilities),
                       (utilities > means).astype(np.uint8))


@given(utilities=utilities(min_cands=1, max_cands=8, min_voters=1,
                           max_voters=20))
def test_honest_rankings_rows_are_permutations(utilities):
    """Each honest ranking must be a permutation of all candidate IDs."""
    rankings = honest_rankings(utilities)
    n_cands = utilities.shape[1]

    for row in rankings:
        assert sorted(row.tolist()) == list(range(n_cands))


@given(utilities=utilities(min_cands=1, max_cands=8, min_voters=1,
                           max_voters=20),
       max_score=integers(1, 50))
def test_honest_normed_scores_use_the_full_scale(utilities, max_score):
    """
    Non-degenerate rows span the whole score scale: the least favorite gets 0
    and the favorite gets max_score.
    """
    scores = honest_normed_scores(utilities, max_score)

    assert scores.min() >= 0
    assert scores.max() <= max_score

    varying = utilities.max(axis=1) > utilities.min(axis=1)
    if varying.any():
        assert_array_equal(scores[varying].min(axis=1), 0)
        assert_array_equal(scores[varying].max(axis=1), max_score)



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
