import numpy as np
import pytest

from elsim.methods import three_two_one


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
