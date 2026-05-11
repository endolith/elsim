import numpy as np
import pytest

from elsim.methods.partisan_primaries import (
    closed_partisan_primary_runoff,
    nominee_restricted_plurality,
    open_partisan_primary,
    pairwise_majority_from_rankings,
    top_two_runoff_reduced_turnout,
)


def test_nominee_restricted_plurality_basic():
    rankings = np.array([
        [0, 1, 2, 3],
        [0, 2, 1, 3],
        [2, 3, 0, 1],
        [2, 0, 1, 3],
    ], dtype=np.uint8)
    assert nominee_restricted_plurality(
        rankings, [0, 1], np.array([0, 1]), 'order') == 0
    assert nominee_restricted_plurality(
        rankings, [2, 3], np.array([2, 3]), 'order') == 2


def test_nominee_restricted_plurality_tie_order():
    rankings = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    assert nominee_restricted_plurality(
        rankings, [0, 1], np.array([0, 1]), 'order') == 0


def test_pairwise_majority_prefers_a():
    rankings = np.array([
        [0, 1],
        [0, 1],
        [1, 0],
    ], dtype=np.uint8)
    assert pairwise_majority_from_rankings(
        rankings, [0, 1, 2], 0, 1, 'order') == 0


def test_pairwise_majority_prefers_b():
    rankings = np.array([
        [1, 0],
        [1, 0],
        [0, 1],
    ], dtype=np.uint8)
    assert pairwise_majority_from_rankings(
        rankings, [0, 1, 2], 0, 1, 'order') == 1


def test_pairwise_majority_tie_order():
    rankings = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    assert pairwise_majority_from_rankings(
        rankings, [0, 1], 0, 1, 'order') == 0


def test_open_partisan_primary():
    rankings = np.array([
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [2, 3, 0, 1],
        [2, 3, 0, 1],
    ], dtype=np.uint8)
    w = open_partisan_primary(
        rankings, 2, [0, 1], [2, 3], [0, 1, 2, 3], 'order')
    assert w == 0


def test_closed_partisan_primary_runoff_subset():
    rankings = np.array([
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [2, 3, 0, 1],
        [2, 3, 0, 1],
    ], dtype=np.uint8)
    w = closed_partisan_primary_runoff(
        rankings, 2, [0, 1], [2, 3], [0, 2], 'order')
    assert w == 0


def test_top_two_runoff_unanimous_first_place():
    rankings = np.array([[0, 1, 2], [0, 2, 1], [0, 1, 2]], dtype=np.uint8)
    w = top_two_runoff_reduced_turnout(
        rankings, [0, 1, 2], [0, 1, 2], 'order')
    assert w == 0


def test_top_two_runoff_two_finalists_pairwise():
    rankings = np.array([
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
    ], dtype=np.uint8)
    w = top_two_runoff_reduced_turnout(
        rankings, np.arange(4), np.arange(4), 'order')
    assert w == 0


def test_top_two_runoff_sntv_returns_none():
    rankings = np.array([
        [0, 1, 2, 3],
        [1, 0, 2, 3],
        [2, 1, 0, 3],
        [3, 1, 2, 0],
    ], dtype=np.uint8)
    assert top_two_runoff_reduced_turnout(
        rankings, [0, 1, 2, 3], [0, 1, 2, 3], tiebreaker=None) is None


def test_top_two_runoff_too_many_sntv_winners_returns_none():
    from unittest.mock import patch

    rankings = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    with patch('elsim.methods.partisan_primaries.sntv',
               return_value={0, 1, 2}):
        assert top_two_runoff_reduced_turnout(
            rankings, [0, 1], [0, 1], 'order') is None
