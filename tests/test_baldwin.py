import numpy as np

from elsim.methods import baldwin, condorcet, total_vote_runoff


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
    """The Baldwin and TVR aliases should return the same winner."""
    election = _example_election()

    assert total_vote_runoff(election) == baldwin(election)


def test_baldwin_elects_a_first_choice_majority_winner():
    """A first-choice majority candidate must win: it is the Condorcet winner
    and Baldwin stops as soon as a candidate holds a first-choice majority."""
    a, b, c = 0, 1, 2
    election = np.array([
        [a, b, c],
        [a, b, c],
        [a, c, b],
        [b, c, a],
        [c, b, a],
    ])

    assert baldwin(election) == a


def test_baldwin_returns_none_for_an_unbroken_score_tie():
    """No elimination may be implied when tied low scores are unresolved."""
    election = np.array([
        [0, 1, 2],
        [1, 2, 0],
        [2, 0, 1],
    ])

    assert baldwin(election) is None


def test_baldwin_order_tiebreak_produces_a_winner():
    """The order rule must resolve a tied lowest score deterministically."""
    election = np.array([
        [0, 1, 2],
        [1, 2, 0],
        [2, 0, 1],
    ])

    assert baldwin(election, tiebreaker='order') == 0


def test_baldwin_reaches_a_final_candidate_without_a_majority():
    """The count must return the last candidate after an unresolved final tie."""
    election = np.array([
        [0, 1, 2],
        [0, 1, 2],
        [1, 0, 2],
        [1, 0, 2],
    ])

    assert baldwin(election, tiebreaker='order') == 0
