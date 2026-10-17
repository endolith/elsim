"""
Shared machinery for sequential-elimination ("runoff") ranked methods.

IRV and Baldwin's method (TVR) differ only in *which* candidate is
eliminated each round.  Everything else -- tallying first preferences,
checking for a majority, recording round data, advancing ballots past
eliminated candidates, and building the result dict -- lives here so that
each method module only has to supply an elimination rule.
"""

import numpy as np

from elsim.methods._common import _inc_rank_idx, _tally_at_rank_idx


def top_choices(election, voter_top_rank_idx):
    """Return each voter's highest-ranked candidate who is still remaining."""
    return election[np.arange(len(election)), voter_top_rank_idx]


def majority_winner(tallies_list, n_voters):
    """Return the candidate holding a strict majority, or ``None``."""
    top = max(tallies_list)
    if top > n_voters / 2:
        return tallies_list.index(top)
    return None


def remaining_minimum(values_list, eliminated_mask):
    """
    Return the candidates that share the lowest value among those remaining.

    Eliminated candidates are ignored entirely, so a remaining candidate
    with a value of 0 is still correctly identified as last place.
    """
    remaining = [c for c, gone in enumerate(eliminated_mask.tolist())
                 if not gone]
    lowest = min(values_list[c] for c in remaining)
    return [c for c in remaining if values_list[c] == lowest]


def run_elimination_rounds(election, choose_losers, tiebreak, *,
                           min_remaining=1, record_rounds=False,
                           round_extras=None, eliminated_mask=None):
    """
    Eliminate one candidate per round until a majority winner appears or
    ``min_remaining`` candidates are left.

    Parameters
    ----------
    election : ndarray
        Ranked ballots (n_voters × n_cands).
    choose_losers : callable
        ``choose_losers(election, eliminated_mask, top_tallies)`` returning
        ``(tied_candidates, before_extras)`` where ``tied_candidates`` is a
        list of candidate IDs tied for elimination and ``before_extras`` is a
        dict merged into the round record (only when ``record_rounds``).
    tiebreak : callable
        Tie-breaking function from ``_get_tiebreak``.
    min_remaining : int, optional
        Stop when this many candidates remain (default 1).
    record_rounds : bool, optional
        If True, each round record includes ``ballots_before``,
        ``ballots_after`` and ``affected_voters`` plus any extras.
    round_extras : callable, optional
        ``round_extras(election, eliminated_mask, loser)`` called *after*
        the loser is eliminated; its dict is merged into the round record
        (only when ``record_rounds``).
    eliminated_mask : ndarray of bool, optional
        Candidates already eliminated before the first round.

    Returns
    -------
    result : dict or None
        ``None`` on an unbroken elimination tie, otherwise the dict shared by
        `irv_rounds` and `baldwin_rounds` (see their docstrings).
    """
    election = np.asarray(election)
    n_voters, n_cands = election.shape
    voter_top_rank_idx = np.zeros(n_voters, dtype=np.uint8)
    top_tallies = np.empty(n_cands, dtype=np.uint)

    if eliminated_mask is None:
        eliminated_mask = np.zeros(n_cands, dtype=bool)
    else:
        eliminated_mask = np.array(eliminated_mask, dtype=bool)  # own copy
        if eliminated_mask.any():
            _inc_rank_idx(election, voter_top_rank_idx, eliminated_mask)

    rounds = []
    winner = None

    while np.count_nonzero(~eliminated_mask) > min_remaining:
        _tally_at_rank_idx(top_tallies, election, voter_top_rank_idx)
        # (tolist makes the pure-Python parts 2-4x faster)
        winner = majority_winner(top_tallies.tolist(), n_voters)
        if winner is not None:
            break

        tied, before_extras = choose_losers(election, eliminated_mask,
                                            top_tallies)
        loser = tiebreak(tied)[0]
        if loser is None:
            return None

        if record_rounds:
            ballots_before = top_choices(election, voter_top_rank_idx).copy()

        eliminated_mask[loser] = True
        _inc_rank_idx(election, voter_top_rank_idx, eliminated_mask)

        round_data = {'loser': loser}
        if record_rounds:
            ballots_after = top_choices(election, voter_top_rank_idx).copy()
            round_data.update(
                ballots_before=ballots_before,
                ballots_after=ballots_after,
                affected_voters=np.flatnonzero(ballots_before == loser),
                **before_extras,
            )
            if round_extras is not None:
                round_data.update(
                    round_extras(election, eliminated_mask, loser))
        rounds.append(round_data)

    if winner is None:
        remaining = np.flatnonzero(~eliminated_mask)
        if len(remaining) == 1:
            winner = int(remaining[0])

    _tally_at_rank_idx(top_tallies, election, voter_top_rank_idx)
    return {
        'winner': winner,
        'rounds': rounds,
        'eliminated_mask': eliminated_mask,
        'final_ballots': top_choices(election, voter_top_rank_idx).copy(),
        'final_tallies': top_tallies.copy(),
    }


def winner_from_result(result, method_name):
    """Unwrap a ``*_rounds`` result into a winner ID for the simple API."""
    if result is None:
        return None
    if result['winner'] is None:
        raise RuntimeError(f'Bug in {method_name} calculation')
    return result['winner']
