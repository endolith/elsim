"""Tests for the combined IRV and Baldwin example driver."""

import numpy as np

from examples.collapse_finder_2d_both import election_to_traces
from elsim.methods import condorcet


def test_election_to_traces_returns_both_method_traces():
    """The driver should derive compatible IRV and Baldwin traces once."""
    candidates = np.array([
        [-1.0, 0.0],
        [0.0, 0.0],
        [0.5, 1.0],
    ])
    voters = np.array([
        [-1.0, 0.0],
        [-1.0, 0.0],
        [0.5, 1.0],
        [0.5, 1.0],
        [0.0, 0.0],
    ])

    rankings, irv_trace, tvr_trace = election_to_traces(voters, candidates)

    assert rankings.shape == (5, 3)
    assert irv_trace is not None
    assert tvr_trace is not None
    assert condorcet(rankings) == 1
    assert irv_trace.rounds[0].eliminated == 1
    assert tvr_trace.winner == 1
