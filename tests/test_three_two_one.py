import numpy as np
import pytest

from elsim.methods import three_two_one
from elsim.methods.three_two_one import _pairwise_rating_preference


def test_three_two_one_invalid_ballots():
    with pytest.raises(ValueError):
        three_two_one([[3, 0]])


def test_three_two_one_invalid_tiebreaker():
    with pytest.raises(ValueError):
        three_two_one([[2, 1, 0]], tiebreaker='nope')


def test_three_two_one_single_candidate():
    election = np.array([[2], [1], [0]], dtype=np.uint8)
    assert three_two_one(election, 'order') == 0


def test_three_two_one_two_candidates():
    election = np.array([[2, 0], [2, 0], [0, 2]], dtype=np.uint8)
    assert three_two_one(election, 'order') == 0


def test_three_two_one_two_candidates_second_wins():
    election = np.array([[0, 2], [0, 2], [2, 0]], dtype=np.uint8)
    assert three_two_one(election, 'order') == 1


def test_three_two_one_requires_2d():
    with pytest.raises(ValueError, match='2D'):
        three_two_one(np.array([0, 1, 2]))


def test_pairwise_rating_preference_total_score_favors_a():
    election = np.array([[0, 1], [2, 0]], dtype=np.uint8)
    assert _pairwise_rating_preference(election, 0, 1, 'order') == 0


def test_pairwise_rating_preference_total_score_favors_b():
    election = np.array([[0, 2], [1, 0]], dtype=np.uint8)
    assert _pairwise_rating_preference(election, 0, 1, 'order') == 1


def test_pairwise_rating_preference_full_tie_none_tiebreaker():
    election = np.array([[1, 1], [1, 1]], dtype=np.uint8)
    assert _pairwise_rating_preference(election, 0, 1, None) is None


def test_three_two_one_eliminates_most_bad():
    # Semifinalists 0,1,2 by Good count; among them candidate 1 has most Bad.
    election = np.array([
        [2, 1, 0],
        [2, 1, 0],
        [2, 0, 1],
        [2, 0, 1],
        [0, 2, 1],
    ], dtype=np.uint8)
    assert three_two_one(election, 'order') == 0
