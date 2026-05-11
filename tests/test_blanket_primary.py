import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import integers, lists, permutations

from elsim.methods import (approval_runoff, irv, irv_eliminate_to_n,
                           irv_primary_top_n_irv, irv_primary_top_n_runoff,
                           runoff, top_five_condorcet, top_four_irv,
                           top_four_runoff, top_five_irv, top_n_condorcet,
                           top_n_irv, top_n_runoff)
from elsim.methods.blanket_primary import (_head_to_head_two,
                                           _top_n_from_plurality_tallies)


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


def test_irv_eliminate_to_n_invalid_n():
    with pytest.raises(ValueError, match='n must be at least 1'):
        irv_eliminate_to_n([[0, 1]], 0)


def test_irv_eliminate_to_n_all_survive_when_n_ge_total():
    assert irv_eliminate_to_n(np.array([[0, 1], [1, 0]]), 2, 'order') == {0, 1}


def test_irv_eliminate_to_n_pre_elimination_zero_first_place():
    election = np.array([[0, 1, 2, 3],
                         [0, 1, 2, 3],
                         [1, 0, 2, 3],
                         [1, 0, 2, 3]])
    assert irv_eliminate_to_n(election, 2, tiebreaker='order') == {0, 1}


def test__top_n_from_plurality_tallies():
    with pytest.raises(ValueError, match='n must be at least 1'):
        _top_n_from_plurality_tallies(np.array([1, 2]), 0, None)
    assert _top_n_from_plurality_tallies(np.array([5, 4]), 2, None) == {0, 1}
    assert _top_n_from_plurality_tallies(np.array([9, 4, 2, 1]), 2, None) == {0, 1}
    assert _top_n_from_plurality_tallies(np.array([5, 5, 5, 5]), 2, None) is None


def test__head_to_head_two():
    e2 = np.array([[0, 1], [0, 1], [1, 0]])
    assert _head_to_head_two(0, 1, e2, tiebreaker='order') == 0
    assert _head_to_head_two(0, 1, np.array([[1, 0], [1, 0], [1, 0]]),
                             tiebreaker='order') == 1
    assert _head_to_head_two(0, 1, np.array([[0, 1], [1, 0]]),
                             tiebreaker=None) is None


def test_approval_runoff_primary_and_general_edges():
    app = np.ones((4, 4), dtype=np.uint8)
    ranked = np.array([[0, 1, 2, 3]] * 4)
    assert approval_runoff(app, ranked, tiebreaker=None) is None

    app_one = np.array([[1], [1]], dtype=np.uint8)
    ranked_one = np.array([[0], [0]])
    assert approval_runoff(app_one, ranked_one) == 0

    with pytest.raises(ValueError, match='0 and 1'):
        approval_runoff(np.array([[2, 0]], dtype=np.uint8),
                         np.array([[0, 1]]))

    app_tie = np.array([[1, 1], [1, 1]], dtype=np.uint8)
    ranked_tie = np.array([[0, 1], [1, 0]])
    assert approval_runoff(app_tie, ranked_tie, tiebreaker=None) is None


def test_top_n_irv_primary_tie_returns_none():
    first = np.array([0, 0, 0, 1, 1, 2, 2, 3])
    rows = []
    for fp in first:
        tail = [c for c in range(4) if c != fp]
        rows.append([fp, *tail])
    election = np.array(rows)
    assert top_n_irv(election, 2, tiebreaker=None) is None


def test_top_n_irv_general_returns_none():
    election = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]])
    assert top_n_irv(election, 3, tiebreaker=None) is None


def test_top_n_runoff_returns_none_paths():
    first = np.array([0, 0, 0, 1, 1, 2, 2, 3])
    rows = []
    for fp in first:
        tail = [c for c in range(4) if c != fp]
        rows.append([fp, *tail])
    election = np.array(rows)
    assert top_n_runoff(election, 2, tiebreaker=None) is None

    election = np.array([[2, 0, 1],
                         [0, 1, 2],
                         [1, 0, 2],
                         [2, 0, 1],
                         [2, 1, 0],
                         [0, 1, 2],
                         [2, 0, 1],
                         [1, 0, 2],
                         [2, 1, 0],
                         [0, 2, 1]])
    assert runoff(election) is None
    assert top_n_runoff(election, 3, None) is None


def test_top_n_condorcet_primary_tie_returns_none():
    first = np.array([0, 0, 0, 1, 1, 2, 2, 3])
    rows = [[fp, *[c for c in range(4) if c != fp]] for fp in first]
    election = np.array(rows)
    assert top_n_condorcet(election, 2, tiebreaker=None) is None


def test_top_n_condorcet_returns_none_on_cycle():
    election = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]])
    assert top_n_condorcet(election, 3, tiebreaker='order') is None


def test_top_five_condorcet_smoke():
    assert top_five_condorcet([[0, 1], [0, 1], [1, 0]], tiebreaker='order') == 0


def test_irv_primary_top_n_returns_none():
    assert irv_primary_top_n_irv([[0, 1, 2], [1, 2, 0], [2, 0, 1]], 2,
                                 tiebreaker=None) is None
    assert irv_primary_top_n_irv([[0, 1, 2], [1, 2, 0], [2, 0, 1]], 3,
                                 tiebreaker=None) is None
    assert irv_primary_top_n_runoff([[0, 1, 2], [1, 2, 0], [2, 0, 1]], 2,
                                      tiebreaker=None) is None

    election = np.array([[2, 0, 1],
                         [0, 1, 2],
                         [1, 0, 2],
                         [2, 0, 1],
                         [2, 1, 0],
                         [0, 1, 2],
                         [2, 0, 1],
                         [1, 0, 2],
                         [2, 1, 0],
                         [0, 2, 1]])
    assert irv_primary_top_n_runoff(election, 3, None) is None


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
