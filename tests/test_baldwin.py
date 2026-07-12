import numpy as np

from elsim.methods import (
    baldwin,
    baldwin_rounds,
    condorcet,
    total_vote_runoff,
    total_vote_runoff_rounds,
)


def _example_election():
    """Return an election where Baldwin and IRV choose different winners."""
    a, b, c = 0, 1, 2
    return np.array([
        [a, c, b],
        [a, c, b],
        [b, c, a],
        [b, c, a],
        [c, a, b],
    ])


def test_baldwin_elects_the_condorcet_winner():
    """The method must retain and elect an existing Condorcet winner."""
    election = _example_election()

    assert condorcet(election) == 2
    assert baldwin(election) == 2


def test_total_vote_runoff_elects_the_baldwin_winner():
    """The majority shortcut may shorten counting but must not change winner."""
    election = _example_election()

    assert total_vote_runoff(election) == baldwin(election)


def test_total_vote_runoff_stops_at_a_new_majority():
    """TVR must stop after a transfer creates a first-choice majority."""
    election = _example_election()

    baldwin_result = baldwin_rounds(election)
    tvr_result = total_vote_runoff_rounds(election)

    assert len(baldwin_result.rounds) == 2
    assert len(tvr_result.rounds) == 1
    assert baldwin_result.winner == tvr_result.winner == 2


def test_baldwin_round_records_exact_score_transition():
    """Per-voter score changes must reproduce the retallied Borda scores."""
    result = baldwin_rounds(_example_election())
    round_ = result.rounds[0]

    assert round_.eliminated == 1
    np.testing.assert_array_equal(round_.borda_before, [10, 9, 11])
    np.testing.assert_array_equal(round_.borda_after, [7, 0, 8])

    running_scores = round_.borda_before.copy()
    n_active = np.count_nonzero(round_.borda_before)
    for higher_ranked in round_.higher_ranked_candidates:
        eliminated_points = n_active - len(higher_ranked)
        running_scores[round_.eliminated] -= eliminated_points
        running_scores[higher_ranked] -= 1

    np.testing.assert_array_equal(running_scores, round_.borda_after)


def test_baldwin_round_records_first_choice_transfers():
    """The trace must identify voters whose visible support changes."""
    result = total_vote_runoff_rounds(_example_election())
    round_ = result.rounds[0]

    np.testing.assert_array_equal(round_.first_tallies_before, [2, 2, 1])
    np.testing.assert_array_equal(round_.first_tallies_after, [2, 0, 3])
    np.testing.assert_array_equal(round_.transferred_voters, [2, 3])
    np.testing.assert_array_equal(round_.transferred_to, [2, 2])


def test_baldwin_rounds_can_stop_with_two_candidates():
    """Analysis callers must be able to inspect a pre-final-round state."""
    result = baldwin_rounds(_example_election(), stop_at=2)

    assert result.winner is None
    np.testing.assert_array_equal(result.active_candidates, [0, 2])
    assert len(result.rounds) == 1


def test_baldwin_returns_none_for_an_unbroken_score_tie():
    """No elimination may be implied when tied low scores are unresolved."""
    election = np.array([
        [0, 1, 2],
        [1, 2, 0],
        [2, 0, 1],
    ])

    assert baldwin(election) is None
    assert baldwin_rounds(election) is None


def test_baldwin_order_tiebreak_produces_a_winner():
    """The order rule must resolve a tied lowest score deterministically."""
    election = np.array([
        [0, 1, 2],
        [1, 2, 0],
        [2, 0, 1],
    ])

    assert baldwin(election, tiebreaker='order') == 0
