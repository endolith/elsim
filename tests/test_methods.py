import numpy as np
import pytest
from _helpers import permute_ranked_candidates
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.strategies import integers
from test_condorcet import complete_ranked_ballots

from elsim.methods import (
    approval, approval_runoff, black, borda, combined_approval, condorcet,
    coombs, fptp, irv, irv_primary_top_n_irv, irv_primary_top_n_runoff,
    runoff, score, top_four_condorcet, top_four_irv, top_five_irv,
    top_four_runoff, top_five_runoff, top_n_condorcet, top_n_irv,
    top_n_runoff, utility_winner)

_blanket_ranked = [
    top_four_irv, top_five_irv, top_four_runoff, top_five_runoff,
    lambda e, tb=None: irv_primary_top_n_irv(e, 4, tb),
    lambda e, tb=None: irv_primary_top_n_runoff(e, 4, tb),
    lambda e, tb=None: top_n_irv(e, 3, tb),
    lambda e, tb=None: top_n_runoff(e, 3, tb),
    lambda e, tb=None: top_n_condorcet(e, 3, tb),
    top_four_condorcet,
]


@pytest.mark.parametrize("method", [black, borda, fptp, runoff, irv, coombs,
                                    approval, combined_approval,
                                    utility_winner, score])
def test_invalid_tiebreaker(method):
    election = [[0, 1],
                [1, 0]]
    with pytest.raises(ValueError):
        method(election, tiebreaker='duel')


def test_invalid_tiebreaker_approval_runoff():
    with pytest.raises(ValueError):
        approval_runoff([[1, 0]], [[0, 1]], tiebreaker='duel')


@pytest.mark.parametrize("method", [black, borda, fptp, runoff, irv, coombs,
                                    *_blanket_ranked])
def test_ranked_method_degenerate_case(method):
    election = [[0]]
    assert method(election) == 0
    assert method(election, 'random') == 0
    assert method(election, 'order') == 0

    election = [[0], [0], [0]]
    assert method(election) == 0
    assert method(election, 'random') == 0
    assert method(election, 'order') == 0


@pytest.mark.parametrize("method", [black, borda, fptp, runoff, irv, coombs,
                                    *_blanket_ranked])
def test_ranked_method_unanimity(method):
    election = [[3, 0, 1, 2], [3, 0, 2, 1], [3, 2, 1, 0]]
    assert method(election) == 3
    assert method(election, 'random') == 3
    assert method(election, 'order') == 3


RANKED_METHODS = [black, borda, condorcet, coombs, fptp, irv, runoff]


@pytest.mark.parametrize("method", RANKED_METHODS)
@given(election=complete_ranked_ballots(min_cands=1, max_cands=8,
                                        min_voters=1, max_voters=40),
       seed=integers(0, 2**31 - 1))
def test_ranked_methods_are_candidate_neutral(method, election, seed):
    """
    Permuting candidate IDs permutes the winner and preserves "no winner".

    This catches implementations that confuse a candidate's position inside a
    subset with its actual ID.
    """
    election = np.asarray(election)
    permutation = np.random.default_rng(seed).permutation(
        election.shape[1])

    winner = method(election)
    permuted_winner = method(
        permute_ranked_candidates(election, permutation))

    if winner is None:
        assert permuted_winner is None
    else:
        assert permuted_winner == permutation[winner]


@pytest.mark.parametrize("method", RANKED_METHODS)
@given(election=complete_ranked_ballots(min_cands=1, max_cands=8,
                                        min_voters=1, max_voters=40),
       seed=integers(0, 2**31 - 1))
def test_ranked_methods_are_voter_anonym(method, election, seed):
    """Reordering the voters must not change a ranked method's winner."""
    election = np.asarray(election)
    order = np.random.default_rng(seed).permutation(len(election))

    assert method(election) == method(election[order])


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

    assert method(election) == expected


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
        assert method(election) == winner


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
