import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import integers, lists, permutations

from elsim.methods import (approval_runoff, irv, irv_eliminate_to_n,
                           irv_primary_top_n_irv, irv_primary_top_n_runoff,
                           runoff, top_four_irv, top_four_runoff, top_five_irv,
                           top_n_condorcet, top_n_irv, top_n_runoff)


def complete_ranked_ballots(min_cands=3, max_cands=256, min_voters=1,
                            max_voters=1000):
    n_cands = integers(min_value=min_cands, max_value=max_cands)
    return n_cands.flatmap(lambda n: lists(permutations(range(n)),
                                           min_size=min_voters,
                                           max_size=max_voters))


@pytest.mark.parametrize('tiebreaker', [None, 'random', 'order'])
def test_top_four_irv_tennessee(tiebreaker):
    Memphis, Nashville, Chattanooga, Knoxville = 0, 1, 2, 3
    election = [*42 * [[Memphis, Nashville, Chattanooga, Knoxville]],
                *26 * [[Nashville, Chattanooga, Knoxville, Memphis]],
                *15 * [[Chattanooga, Knoxville, Nashville, Memphis]],
                *17 * [[Knoxville, Chattanooga, Nashville, Memphis]],
                ]
    assert top_four_irv(election, tiebreaker) == Knoxville
    assert top_five_irv(election, tiebreaker) == Knoxville
    assert irv(election, tiebreaker) == Knoxville


@pytest.mark.parametrize('tiebreaker', [None, 'random', 'order'])
def test_top_four_runoff_tennessee(tiebreaker):
    Memphis, Nashville, Chattanooga, Knoxville = 0, 1, 2, 3
    election = [*42 * [[Memphis, Nashville, Chattanooga, Knoxville]],
                *26 * [[Nashville, Chattanooga, Knoxville, Memphis]],
                *15 * [[Chattanooga, Knoxville, Nashville, Memphis]],
                *17 * [[Knoxville, Chattanooga, Nashville, Memphis]],
                ]
    assert top_four_runoff(election, tiebreaker) == runoff(
        election, tiebreaker)


@pytest.mark.parametrize('tiebreaker', [None, 'random', 'order'])
def test_irv_primary_variants_tennessee(tiebreaker):
    Memphis, Nashville, Chattanooga, Knoxville = 0, 1, 2, 3
    election = [*42 * [[Memphis, Nashville, Chattanooga, Knoxville]],
                *26 * [[Nashville, Chattanooga, Knoxville, Memphis]],
                *15 * [[Chattanooga, Knoxville, Nashville, Memphis]],
                *17 * [[Knoxville, Chattanooga, Nashville, Memphis]],
                ]
    assert irv_primary_top_n_irv(election, 4, tiebreaker) == Knoxville
    assert irv_primary_top_n_runoff(election, 4, tiebreaker) == runoff(
        election, tiebreaker)


def test_irv_eliminate_to_n_three_cycle():
    election = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]])
    assert irv_eliminate_to_n(election, 2) is None
    assert sorted(irv_eliminate_to_n(election, 2, tiebreaker='order')) == [0, 1]


def test_approval_runoff_matches_head_to_head():
    A, B, C = 0, 1, 2
    approvals = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=np.uint8)
    ranked = np.array([[A, B, C], [B, A, C], [C, B, A]])
    assert approval_runoff(approvals, ranked, tiebreaker='order') == 1


def test_approval_runoff_row_mismatch():
    with pytest.raises(ValueError, match='same number of rows'):
        approval_runoff([[1, 0]], [[0, 1], [1, 0]])


def test_blanket_methods_reject_invalid_tiebreaker():
    election = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
    with pytest.raises(ValueError):
        top_n_irv(election, 2, tiebreaker='duel')
    with pytest.raises(ValueError):
        top_n_runoff(election, 2, tiebreaker='duel')
    with pytest.raises(ValueError):
        top_n_condorcet(election, 2, tiebreaker='duel')
    with pytest.raises(ValueError):
        irv_eliminate_to_n(election, 2, tiebreaker='duel')


@pytest.mark.parametrize('tiebreaker', ['random', 'order'])
@given(election=complete_ranked_ballots(min_cands=2, max_cands=25,
                                        min_voters=1, max_voters=100))
def test_top_n_irv_winner_in_range(election, tiebreaker):
    n_cands = np.shape(election)[1]
    w = top_n_irv(election, min(4, n_cands), tiebreaker)
    assert w in range(n_cands)


@given(election=complete_ranked_ballots(min_cands=2, max_cands=25,
                                        min_voters=1, max_voters=100))
def test_top_n_irv_winner_none_tiebreaker(election):
    n_cands = np.shape(election)[1]
    w = top_n_irv(election, min(4, n_cands))
    assert w in {None} | set(range(n_cands))


if __name__ == '__main__':
    from subprocess import PIPE, Popen
    with Popen(['pytest', '--tb=short', str(__file__)],
               stdout=PIPE, bufsize=1, universal_newlines=True) as p:
        for line in p.stdout:
            print(line, end='')
