import numpy as np
import pytest

from elsim.methods.irv import irv, irv_rounds


def test_irv_rounds_records_first_choice_transfers():
    """The trace must identify exactly which ballots move after elimination."""
    a, b, c = 0, 1, 2
    election = np.array([
        [a, c, b],
        [a, c, b],
        [b, c, a],
        [b, c, a],
        [c, a, b],
    ])

    result = irv_rounds(election)

    assert result.winner == a
    assert len(result.rounds) == 1
    round_ = result.rounds[0]
    assert round_.eliminated == c
    np.testing.assert_array_equal(round_.tallies_before, [2, 2, 1])
    np.testing.assert_array_equal(round_.tallies_after, [3, 2, 0])
    np.testing.assert_array_equal(round_.transferred_voters, [4])
    np.testing.assert_array_equal(round_.transferred_to, [a])


def test_irv_rounds_exposes_initial_zero_vote_eliminations():
    """Zero-vote exclusions must be visible without inventing a tie order."""
    a, b, c, d = 0, 1, 2, 3
    election = np.array([
        [a, b, c, d],
        [a, b, d, c],
        [b, a, c, d],
    ])

    result = irv_rounds(election)

    np.testing.assert_array_equal(result.initially_eliminated, [c, d])
    assert result.winner == a
    assert result.rounds == ()


def test_irv_rounds_can_stop_with_two_candidates():
    """Animation callers need a complete trace through the final-two state."""
    a, b, c = 0, 1, 2
    election = np.array([
        [a, b, c],
        [a, c, b],
        [b, c, a],
        [b, c, a],
        [c, a, b],
    ])

    result = irv_rounds(election, stop_at=2)

    assert result.winner is None
    np.testing.assert_array_equal(result.active_candidates, [a, b])
    np.testing.assert_array_equal(result.final_tallies, [3, 2, 0])


def test_irv_rounds_returns_none_for_an_unbroken_elimination_tie():
    """A trace must not imply a transfer when no tie-breaking rule was given."""
    election = np.array([
        [0, 1, 2],
        [1, 2, 0],
        [2, 0, 1],
    ])

    assert irv_rounds(election) is None


@pytest.mark.parametrize('stop_at', [0, 4])
def test_irv_rounds_rejects_out_of_range_stop_counts(stop_at):
    """Invalid stopping counts would otherwise create incomplete trace states."""
    election = np.array([
        [0, 1, 2],
        [1, 2, 0],
    ])

    with pytest.raises(ValueError, match='stop_at must be between'):
        irv_rounds(election, stop_at=stop_at)


def test_irv_rounds_rejects_non_integer_stop_counts():
    """A fractional stopping count has no meaningful election interpretation."""
    election = np.array([
        [0, 1],
        [1, 0],
    ])

    with pytest.raises(TypeError, match='stop_at must be an integer'):
        irv_rounds(election, stop_at=1.5)


@pytest.mark.parametrize('tiebreaker', [None, 'order'])
def test_irv_rounds_matches_the_existing_winner_api(tiebreaker):
    """Adding traces must not change the winner returned by ordinary IRV."""
    election = np.array([
        [0, 2, 1],
        [0, 2, 1],
        [1, 2, 0],
        [1, 2, 0],
        [2, 0, 1],
    ])

    result = irv_rounds(election, tiebreaker=tiebreaker)
    traced_winner = None if result is None else result.winner

    assert traced_winner == irv(election, tiebreaker=tiebreaker)
