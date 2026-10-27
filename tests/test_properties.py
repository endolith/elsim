"""Cross-cutting property-based tests for election methods.

These tests use metamorphic properties (relabeling candidates, reordering
voters, comparing equivalent representations) instead of hand-computed
expected winners.  They complement the per-method tests, which mostly check
that a winner is in range.
"""
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats, integers, tuples
from numpy.testing import assert_allclose, assert_array_equal

from elsim.elections import normal_electorate, normed_dist_utilities
from elsim.methods import (approval, black, borda, combined_approval,
                           condorcet, coombs, fptp, irv, runoff, score, sntv,
                           star)
from elsim.methods.condorcet import (condorcet_from_matrix,
                                     ranked_election_to_matrix)
from elsim.strategies import (approval_optimal, honest_normed_scores,
                              honest_rankings, vote_for_k)
from test_approval import approval_ballots, combined_approval_ballots
from test_condorcet import complete_ranked_ballots
from test_score import score_ballots
from test_strategies import utilities as utility_arrays

RANKED_METHODS = [black, borda, condorcet, coombs, fptp, irv, runoff, sntv]


def _winner(method, election):
    """Return a single winner ID, unwrapping SNTV's one-element set."""
    winner = method(election)
    if method is sntv and winner is not None:
        (winner,) = winner
    return winner


def _relabel_ranked(election, perm):
    """Apply candidate relabeling `perm` to ranked ballots."""
    return np.asarray(perm)[np.asarray(election)]


def _relabel_columns(election, perm):
    """Apply candidate relabeling `perm` to candidate-per-column ballots."""
    return np.asarray(election)[:, np.argsort(perm)]


def integer_utilities(min_cands=2, max_cands=8, min_voters=1,
                      max_voters=20):
    """Strategy for small integer-valued utilities, to hit exact means."""
    n_cands = integers(min_value=min_cands, max_value=max_cands)
    n_voters = integers(min_value=min_voters, max_value=max_voters)
    return arrays(np.int64, tuples(n_voters, n_cands),
                  elements=integers(0, 5)).map(lambda a: a.astype(float))


@pytest.mark.parametrize("method", RANKED_METHODS)
@given(election=complete_ranked_ballots(min_cands=1, max_cands=8,
                                        min_voters=1, max_voters=40),
       seed=integers(0, 2**31 - 1))
def test_ranked_methods_are_candidate_neutral(method, election, seed):
    """
    Relabeling the candidates permutes the winner and preserves "no winner".

    This catches implementations that confuse a candidate's position inside a
    subset with its actual ID.
    """
    election = np.asarray(election)
    perm = np.random.default_rng(seed).permutation(election.shape[1])

    winner = _winner(method, election)
    relabeled_winner = _winner(method, _relabel_ranked(election, perm))

    if winner is None:
        assert relabeled_winner is None
    else:
        assert relabeled_winner == perm[winner]


@pytest.mark.parametrize("method", RANKED_METHODS)
@given(election=complete_ranked_ballots(min_cands=1, max_cands=8,
                                        min_voters=1, max_voters=40),
       seed=integers(0, 2**31 - 1))
def test_ranked_methods_are_voter_anonym(method, election, seed):
    """Reordering the voters must not change a ranked method's winner."""
    election = np.asarray(election)
    order = np.random.default_rng(seed).permutation(len(election))

    assert _winner(method, election) == _winner(method, election[order])


@pytest.mark.parametrize("method", RANKED_METHODS)
@given(election=complete_ranked_ballots(min_cands=2, max_cands=2,
                                        min_voters=1, max_voters=40))
def test_two_candidate_ranked_methods_pick_the_majority(method, election):
    """
    With two candidates, every ranked method reduces to majority rule.

    This gives a reference result that does not depend on any tally internals.
    """
    election = np.asarray(election)
    counts = np.bincount(election[:, 0], minlength=2)
    expected = None if counts[0] == counts[1] else int(np.argmax(counts))

    assert _winner(method, election) == expected


@given(data=st.data())
def test_ranked_methods_are_unanimous(data):
    """
    If every voter ranks the same candidate first, every ranked method returns
    that candidate.
    """
    n_cands = data.draw(integers(1, 8), label="n_cands")
    n_voters = data.draw(integers(1, 30), label="n_voters")
    winner = data.draw(integers(0, n_cands - 1), label="winner")
    others = [c for c in range(n_cands) if c != winner]
    rest = data.draw(st.permutations(others), label="rest")
    election = np.array([[winner, *rest]] * n_voters)

    for method in RANKED_METHODS:
        assert _winner(method, election) == winner


@given(election=complete_ranked_ballots(min_cands=1, max_cands=8,
                                        min_voters=1, max_voters=40))
def test_fptp_matches_sntv_on_first_preferences(election):
    """FPTP is SNTV with one winner, whether given 2D or 1D ballots."""
    election = np.asarray(election)
    winner = fptp(election)
    winners = sntv(election[:, 0])

    if winner is None:
        assert winners is None
    else:
        assert winners == {winner}


@given(election=complete_ranked_ballots(min_cands=1, max_cands=8,
                                        min_voters=1, max_voters=40))
def test_condorcet_matches_explicit_matrix_pipeline(election):
    """
    condorcet() must agree with building the pairwise matrix explicitly.

    Hypothesis reuses one process, so this also exercises cache invalidation
    between different same-shaped elections.
    """
    election = np.asarray(election)
    expected = condorcet_from_matrix(ranked_election_to_matrix(election))

    assert condorcet(election) == expected


@pytest.mark.parametrize("method", [score, star])
@given(election=score_ballots(min_cands=1, max_cands=8,
                              min_voters=1, max_voters=40),
       seed=integers(0, 2**31 - 1))
def test_score_methods_are_candidate_neutral(method, election, seed):
    """Relabeling candidates permutes the winner of score-based methods."""
    election = np.asarray(election)
    perm = np.random.default_rng(seed).permutation(election.shape[1])

    winner = method(election)
    relabeled_winner = method(_relabel_columns(election, perm))

    if winner is None:
        assert relabeled_winner is None
    else:
        assert relabeled_winner == perm[winner]


@given(election=score_ballots(min_cands=1, max_cands=8,
                              min_voters=1, max_voters=40))
def test_star_winner_has_a_top_two_score(election):
    """
    A STAR winner is a finalist from the scoring round, so their score is at
    least the second-highest score.

    This catches tiebreak paths that leak subset positions as candidate IDs.
    """
    election = np.asarray(election)
    winner = star(election, tiebreaker='order')
    if winner is None:
        return

    tallies = election.sum(axis=0)
    second_highest = np.sort(tallies)[-2] if len(tallies) > 1 else tallies[0]
    assert tallies[winner] >= second_highest


@pytest.mark.parametrize("method", [approval, combined_approval])
@given(data=st.data())
def test_approval_methods_are_candidate_neutral(method, data):
    """Relabeling candidates permutes the winner of approval-based methods."""
    if method is approval:
        ballots = approval_ballots
    else:
        ballots = combined_approval_ballots
    election = data.draw(ballots(min_cands=1, max_cands=8, min_voters=1,
                                 max_voters=40), label="election")
    perm = data.draw(st.permutations(range(election.shape[1])),
                     label="perm")

    winner = method(election)
    relabeled_winner = method(_relabel_columns(election, perm))

    if winner is None:
        assert relabeled_winner is None
    else:
        assert relabeled_winner == perm[winner]


@given(data=st.data())
def test_vote_for_k_approves_the_top_k_utilities(data):
    """
    vote_for_k approves exactly k candidates per voter, and every approved
    utility is at least every rejected utility.  Negative k is equivalent to
    counting from the other end.
    """
    utilities = data.draw(utility_arrays(min_cands=2, max_cands=8,
                                         min_voters=1, max_voters=20),
                          label="utilities")
    n_cands = utilities.shape[1]
    k = data.draw(integers(1, n_cands - 1), label="k")

    election = vote_for_k(utilities, k)
    assert_array_equal(election, vote_for_k(utilities, k - n_cands))
    assert_array_equal(election.sum(axis=1), k)

    for row, ballot in zip(utilities, election, strict=True):
        approved = row[ballot == 1]
        rejected = row[ballot == 0]
        assert approved.min() >= rejected.max()

    assert_array_equal(vote_for_k(utilities, 'half').sum(axis=1),
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


@given(utilities=utility_arrays(min_cands=1, max_cands=8, min_voters=1,
                                max_voters=20))
def test_honest_rankings_rows_are_permutations(utilities):
    """Each honest ranking must be a permutation of all candidate IDs."""
    rankings = honest_rankings(utilities)
    n_cands = utilities.shape[1]

    for row in rankings:
        assert sorted(row.tolist()) == list(range(n_cands))


@given(utilities=utility_arrays(min_cands=1, max_cands=8, min_voters=1,
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


@given(data=st.data())
def test_normed_dist_utilities_is_bounded_and_affine_invariant(data):
    """
    Utilities stay in [0, 1] and are unchanged when both voter and candidate
    positions are translated and scaled together, since only relative
    distances matter.
    """
    n_voters = data.draw(integers(1, 6), label="n_voters")
    n_cands = data.draw(integers(2, 6), label="n_cands")
    dims = data.draw(integers(1, 4), label="dims")
    voters = data.draw(arrays(np.float64, (n_voters, dims),
                              elements=integers(-3, 3)), label="voters")
    cands = data.draw(arrays(np.float64, (n_cands, dims),
                             elements=integers(-3, 3)), label="cands")
    # Powers of two and integer offsets make the affine transformation exact,
    # so ties between distances are preserved bit-for-bit.
    scale = data.draw(integers(-2, 1).map(lambda e: 2.0 ** e), label="scale")
    offset = data.draw(integers(-3, 3).map(float), label="offset")

    utilities = normed_dist_utilities(voters, cands)
    assert np.all(utilities >= 0)
    assert np.all(utilities <= 1)

    transformed = normed_dist_utilities(voters*scale + offset,
                                        cands*scale + offset)
    assert_allclose(utilities, transformed, rtol=1e-6, atol=1e-9)



@given(dims=integers(2, 5), corr=floats(0, 0.8), disp=floats(0.2, 3),
       seed=integers(0, 2**31 - 1))
def test_normal_electorate_matches_its_correlation_parameters(
        dims, corr, disp, seed):
    """
    corr sets the SD ratio between the first axis and the others, and disp sets
    the candidate/voter SD ratio.  Both are checked on large samples so the
    statistical error is small.
    """
    voters, cands = normal_electorate(10_000, 5_000, dims=dims, corr=corr,
                                      disp=disp, random_state=seed)

    axis_ratio = np.sqrt((1 + (dims - 1)*corr) / (1 - corr))
    assert_allclose(np.std(cands[:, 0]) / np.std(cands[:, 1]), axis_ratio,
                    rtol=0.1)
    assert_allclose(np.std(cands[:, 0]) / np.std(voters[:, 0]), disp,
                    rtol=0.1)
