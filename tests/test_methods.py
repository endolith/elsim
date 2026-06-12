import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import integers, lists, permutations

from elsim.methods import (approval, black, borda, combined_approval, coombs,
                           fptp, irv, runoff, score, utility_winner)


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
