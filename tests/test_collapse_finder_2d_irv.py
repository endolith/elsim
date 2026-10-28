"""Tests for the 2D IRV collapse validator and animation smoke path."""

import matplotlib

matplotlib.use('Agg')
import numpy as np

from examples.collapse_finder_2d_irv import (
    run_irv_animation,
    simulate_irv_rounds,
)
from elsim.methods import condorcet
from elsim.methods.irv import irv_rounds


def _center_outward_election():
    """Return a small election whose center candidate is eliminated first."""
    candidates = np.array([
        [-1.0, 0.0],
        [0.0, 0.0],
        [0.5, 1.0],
    ])
    election = np.array([
        [0, 1, 2],
        [0, 1, 2],
        [2, 1, 0],
        [2, 1, 0],
        [1, 0, 2],
    ])
    return election, candidates


def test_center_outward_validator_accepts_expected_elimination_order():
    """A clean center-first trace should pass geometric validation."""
    election, candidates = _center_outward_election()

    result = simulate_irv_rounds(election, candidates)

    assert result is not None
    assert [round_.eliminated for round_ in result.rounds] == [1]
    np.testing.assert_array_equal(result.active_candidates, [0, 2])
    assert condorcet(election) == result.rounds[0].eliminated


def test_center_outward_validator_rejects_noncenter_elimination():
    """An outer candidate eliminated before the center must be rejected."""
    election, candidates = _center_outward_election()
    noncenter_first = election.copy()
    noncenter_first[0] = [1, 0, 2]

    result = simulate_irv_rounds(noncenter_first, candidates)

    assert result is None


def test_center_outward_validator_rejects_initial_zero_vote_candidates():
    """The validator must reject traces with eager zero-vote exclusions."""
    _, candidates = _center_outward_election()
    election = np.array([
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [1, 0, 2],
    ])

    result = irv_rounds(election, stop_at=2)

    assert result is not None
    assert len(result.initially_eliminated) > 0
    assert simulate_irv_rounds(election, candidates) is None


def test_irv_animation_writes_a_gif_for_a_tiny_election(tmp_path):
    """The renderer should produce a GIF from a small traced election."""
    election, candidates = _center_outward_election()
    trace = simulate_irv_rounds(election, candidates)
    voters = np.array([
        [-1.0, 0.0],
        [-0.5, 0.0],
        [1.0, 0.0],
        [0.5, 0.0],
        [0.0, 0.0],
    ])

    assert trace is not None
    output_dir = run_irv_animation(
        voters,
        candidates,
        election,
        trace,
        tmp_path,
        frames_per_transfer=2,
        seed=7,
    )

    assert (output_dir / 'collapse_2d_irv.gif').is_file()
