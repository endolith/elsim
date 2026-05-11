import numpy as np
import pytest
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats, integers, lists, permutations, tuples

from elsim.methods import (approval, black, borda, combined_approval, coombs,
                           condorcet, fptp, irv, runoff, score, sntv,
                           utility_winner)


@pytest.mark.parametrize("method", [black, borda, fptp, runoff, irv, coombs,
                                    approval, combined_approval,
                                    utility_winner, score])
def test_invalid_tiebreaker(method):
    with pytest.raises(ValueError):
        election = [[0, 1],
                    [1, 0]]
        method(election, tiebreaker='duel')


@pytest.mark.parametrize("method", [black, borda, fptp, runoff, irv, coombs])
def test_ranked_method_degenerate_case(method):
    election = [[0]]
    assert method(election) == 0
    assert method(election, 'random') == 0
    assert method(election, 'order') == 0

    election = [[0], [0], [0]]
    assert method(election) == 0
    assert method(election, 'random') == 0
    assert method(election, 'order') == 0


@pytest.mark.parametrize("method", [black, borda, fptp, runoff, irv, coombs])
def test_ranked_method_unanimity(method):
    election = [[3, 0, 1, 2], [3, 0, 2, 1], [3, 2, 1, 0]]
    assert method(election) == 3
    assert method(election, 'random') == 3
    assert method(election, 'order') == 3


def complete_ranked_ballots(min_cands=2, max_cands=25, min_voters=1,
                            max_voters=100):
    n_cands = integers(min_value=min_cands, max_value=max_cands)
    return n_cands.flatmap(lambda n: lists(permutations(range(n)),
                                           min_size=min_voters,
                                           max_size=max_voters))


@given(election=complete_ranked_ballots(min_cands=2, max_cands=20,
                                        min_voters=1, max_voters=80))
def test_black_matches_condorcet_or_borda(election):
    election = np.asarray(election)
    cw = condorcet(election)
    if cw is not None:
        assert black(election) == cw
    else:
        assert black(election) == borda(election)


@given(election=complete_ranked_ballots(min_cands=2, max_cands=20,
                                        min_voters=1, max_voters=80))
def test_sntv_one_winner_matches_fptp_order(election):
    election = np.asarray(election)
    f_w = fptp(election, tiebreaker='order')
    s_w = sntv(election, 1, tiebreaker='order')
    if f_w is None:
        assert s_w is None
    else:
        assert s_w == {f_w}


@pytest.mark.parametrize(
    "method",
    [black, borda, fptp, runoff, irv, coombs],
)
@given(election=complete_ranked_ballots(min_cands=2, max_cands=15,
                                        min_voters=1, max_voters=60))
def test_ranked_methods_order_tiebreak_returns_candidate_id(method, election):
    election = np.asarray(election)
    winner = method(election, tiebreaker='order')
    n_cands = election.shape[1]
    assert winner in set(range(n_cands))


@pytest.mark.parametrize(
    "method",
    [black, borda, fptp, runoff, irv, coombs],
)
@given(election=complete_ranked_ballots(min_cands=2, max_cands=15,
                                        min_voters=1, max_voters=60))
def test_ranked_methods_no_tiebreak_returns_none_or_id(method, election):
    election = np.asarray(election)
    winner = method(election)
    n_cands = election.shape[1]
    assert winner in {None} | set(range(n_cands))


@given(
    utilities=arrays(
        np.float64,
        tuples(integers(1, 25), integers(2, 12)),
        elements=floats(0, 1, allow_nan=False, allow_infinity=False),
    ),
)
def test_utility_winner_matches_argmax_of_column_sums(utilities):
    winner = utility_winner(utilities, tiebreaker='order')
    totals = utilities.sum(axis=0)
    assert winner == int(np.argmax(totals))


def test_utility_winner_two_way_tie_no_tiebreaker():
    utilities = [[1.0, 0.0], [0.0, 1.0]]
    assert utility_winner(utilities) is None
    assert utility_winner(utilities, tiebreaker='order') == 0


def test_combined_approval_symmetric_tie_no_tiebreaker():
    election = np.array([[1, -1], [-1, 1]], dtype=np.int8)
    assert combined_approval(election) is None
    assert combined_approval(election, tiebreaker='order') == 0


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
