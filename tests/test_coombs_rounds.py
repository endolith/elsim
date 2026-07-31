import numpy as np
import pytest

from elsim.methods.coombs import coombs, coombs_rounds


def test_coombs_rounds_records_last_place_elimination():
    """The trace must retain the last-place tally that caused elimination."""
    a, b, c = 0, 1, 2
    election = np.array([
        [a, c, b],
        [a, c, b],
        [b, c, a],
        [b, c, a],
        [c, a, b],
    ])

    result = coombs_rounds(election)

    assert result.winner == c
    assert len(result.rounds) == 1
    round_ = result.rounds[0]
    assert round_.eliminated == b
    np.testing.assert_array_equal(round_.first_tallies_before, [2, 2, 1])
    np.testing.assert_array_equal(round_.last_tallies_before, [2, 3, 0])
    np.testing.assert_array_equal(round_.first_tallies_after, [2, 0, 3])
    np.testing.assert_array_equal(round_.transferred_voters, [2, 3])
    np.testing.assert_array_equal(round_.transferred_to, [c, c])


def test_coombs_rounds_can_stop_with_two_candidates():
    """A final-two trace must stop without performing an extra elimination."""
    election = np.array([
        [0, 2, 1],
        [0, 2, 1],
        [1, 2, 0],
        [1, 2, 0],
        [2, 0, 1],
    ])

    result = coombs_rounds(election, stop_at=2)

    assert result.winner is None
    np.testing.assert_array_equal(result.active_candidates, [0, 2])
    np.testing.assert_array_equal(result.final_tallies, [2, 0, 3])


def test_coombs_rounds_returns_none_for_an_unbroken_elimination_tie():
    """The trace must stop when tied last-place totals cannot be resolved."""
    election = np.array([
        [0, 1, 2],
        [1, 2, 0],
        [2, 0, 1],
    ])

    assert coombs_rounds(election) is None


@pytest.mark.parametrize('stop_at', [0, 4])
def test_coombs_rounds_rejects_out_of_range_stop_counts(stop_at):
    """Invalid survivor counts would make the final trace state ambiguous."""
    election = np.array([
        [0, 1, 2],
        [1, 2, 0],
    ])

    with pytest.raises(ValueError, match='stop_at must be between'):
        coombs_rounds(election, stop_at=stop_at)


@pytest.mark.parametrize('tiebreaker', [None, 'order'])
def test_coombs_rounds_matches_the_existing_winner_api(tiebreaker):
    """Tracing Coombs rounds must preserve the ordinary winner calculation."""
    election = np.array([
        [0, 2, 1],
        [0, 2, 1],
        [1, 2, 0],
        [1, 2, 0],
        [2, 0, 1],
    ])

    result = coombs_rounds(election, tiebreaker=tiebreaker)
    traced_winner = None if result is None else result.winner

    assert traced_winner == coombs(election, tiebreaker=tiebreaker)
