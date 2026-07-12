"""Tests for the 2D Baldwin collapse validator and animation."""

import matplotlib

matplotlib.use('Agg')
import numpy as np

from examples.collapse_finder_2d_tvr import (
    average_ranks_from_borda,
    run_tvr_animation,
    simulate_tvr_rounds,
)
from elsim.methods.baldwin import baldwin_rounds


def _center_winner_election():
    """Return a small election whose nearest candidate wins Baldwin."""
    candidates = np.array([
        [-1.0, 0.0],
        [0.5, 1.0],
        [0.0, 0.0],
    ])
    election = np.array([
        [0, 2, 1],
        [0, 2, 1],
        [1, 2, 0],
        [1, 2, 0],
        [2, 0, 1],
    ])
    return election, candidates


def test_borda_replay_matches_every_recorded_after_score():
    """The prescribed voter-wise score changes must reproduce Baldwin traces."""
    election, _ = _center_winner_election()
    result = baldwin_rounds(election)

    assert result is not None
    eliminated = set()
    n_cands = election.shape[1]
    for round_ in result.rounds:
        n_active = n_cands - len(eliminated)
        running = round_.borda_before.copy()
        for higher in round_.higher_ranked_candidates:
            running[higher] -= 1
            running[round_.eliminated] -= n_active - len(higher)
        np.testing.assert_array_equal(running, round_.borda_after)
        eliminated.add(round_.eliminated)


def test_average_rank_formula_is_one_based_and_bounded():
    """Borda scores should map best and worst active ranks to 1 and n_active."""
    scores = np.array([12.0, 8.0, 4.0])
    ranks = average_ranks_from_borda(scores, n_active=3, n_voters=4)

    np.testing.assert_allclose(ranks, [1.0, 2.0, 3.0])
    assert np.all((1 <= ranks) & (ranks <= 3))


def test_center_validator_accepts_nearest_winner_and_rejects_other_winner():
    """The validator should test geometric-center convergence only."""
    election, candidates = _center_winner_election()

    assert simulate_tvr_rounds(election, candidates) is not None
    candidates_with_different_center = candidates[[2, 1, 0]]
    assert simulate_tvr_rounds(election, candidates_with_different_center) is None


def test_tvr_animation_writes_a_gif_for_a_tiny_election(tmp_path):
    """The Baldwin renderer should produce a GIF for a small complete trace."""
    election, candidates = _center_winner_election()
    voters = np.array([
        [-1.0, 0.0],
        [-0.5, 0.0],
        [1.0, 0.0],
        [0.5, 0.0],
        [0.0, 0.0],
    ])
    trace = simulate_tvr_rounds(election, candidates)

    assert trace is not None
    output_dir = run_tvr_animation(
        voters,
        candidates,
        election,
        trace,
        tmp_path,
        frames_per_transfer=2,
        seed=7,
    )

    assert (output_dir / 'collapse_2d_tvr.gif').is_file()
