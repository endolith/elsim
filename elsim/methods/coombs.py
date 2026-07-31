from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from elsim.methods._common import (
    _all_indices,
    _dec_rank_idx,
    _get_tiebreak,
    _inc_rank_idx,
    _no_tiebreak,
    _order_tiebreak_elim,
    _random_tiebreak,
    _tally_at_rank_idx,
    _validate_stop_at,
)

_tiebreak_map = {'order': _order_tiebreak_elim,
                 'random': _random_tiebreak,
                 None: _no_tiebreak}


@dataclass(frozen=True)
class CoombsRound:
    """
    One last-place elimination and its first-choice transfers.

    Attributes
    ----------
    eliminated : int
        Candidate removed for receiving the most last-place rankings.
    first_tallies_before, first_tallies_after : ndarray
        First-choice tallies immediately before and after the elimination.
    last_tallies_before : ndarray
        Last-choice tallies used to choose the eliminated candidate.
    transferred_voters : ndarray
        Indices of voters whose first choice was the eliminated candidate.
    transferred_to : ndarray
        New first-choice candidate for each voter in ``transferred_voters``.
    """

    eliminated: int
    first_tallies_before: np.ndarray
    first_tallies_after: np.ndarray
    last_tallies_before: np.ndarray
    transferred_voters: np.ndarray
    transferred_to: np.ndarray


@dataclass(frozen=True)
class CoombsResult:
    """Result and round trace from a Coombs count."""

    winner: Optional[int]
    rounds: Tuple[CoombsRound, ...]
    active_candidates: np.ndarray
    final_choices: np.ndarray
    final_tallies: np.ndarray


def _run_coombs(election, tiebreaker, stop_at, record_rounds):
    """Run the shared Coombs count used by the winner and trace APIs."""
    election = np.asarray(election)
    n_voters, n_cands = election.shape
    stop_at = _validate_stop_at(stop_at, n_cands)
    tiebreak = _get_tiebreak(tiebreaker, _tiebreak_map)
    voter_top_rank_idx = np.zeros(n_voters, dtype=np.intp)
    voter_bottom_rank_idx = np.full(
        n_voters, n_cands - 1, dtype=np.intp
    )
    cand_top_tallies = np.empty(n_cands, dtype=np.uint)
    cand_bottom_tallies = np.empty(n_cands, dtype=np.uint)
    eliminated_mask = np.zeros(n_cands, dtype=bool)
    rounds = []
    winner = None

    while np.count_nonzero(~eliminated_mask) > stop_at:
        _tally_at_rank_idx(
            cand_top_tallies, election, voter_top_rank_idx
        )
        cand_top_tallies_list = cand_top_tallies.tolist()

        max_cand_top_tally = max(cand_top_tallies_list)
        if max_cand_top_tally > n_voters / 2:
            winner = cand_top_tallies_list.index(max_cand_top_tally)
            break

        _tally_at_rank_idx(
            cand_bottom_tallies, election, voter_bottom_rank_idx
        )
        active_bottom_tallies = cand_bottom_tallies[~eliminated_mask]
        max_bottom_tally = int(active_bottom_tallies.max())
        max_bottom_tally_cands = [
            candidate
            for candidate in _all_indices(
                cand_bottom_tallies.tolist(), max_bottom_tally
            )
            if not eliminated_mask[candidate]
        ]
        cand_to_eliminate = tiebreak(max_bottom_tally_cands)[0]
        if cand_to_eliminate is None:
            return None

        if record_rounds:
            choices_before = election[
                np.arange(n_voters), voter_top_rank_idx
            ].copy()
            first_tallies_before = cand_top_tallies.copy()
            last_tallies_before = cand_bottom_tallies.copy()

        eliminated_mask[cand_to_eliminate] = True
        _inc_rank_idx(election, voter_top_rank_idx, eliminated_mask)
        _dec_rank_idx(election, voter_bottom_rank_idx, eliminated_mask)

        if record_rounds:
            choices_after = election[
                np.arange(n_voters), voter_top_rank_idx
            ].copy()
            _tally_at_rank_idx(
                cand_top_tallies, election, voter_top_rank_idx
            )
            transferred_voters = np.flatnonzero(
                choices_before == cand_to_eliminate
            )
            rounds.append(
                CoombsRound(
                    eliminated=int(cand_to_eliminate),
                    first_tallies_before=first_tallies_before,
                    first_tallies_after=cand_top_tallies.copy(),
                    last_tallies_before=last_tallies_before,
                    transferred_voters=transferred_voters,
                    transferred_to=choices_after[transferred_voters],
                )
            )

    active_candidates = np.flatnonzero(~eliminated_mask)
    if winner is None and active_candidates.size == 1:
        winner = int(active_candidates[0])

    _tally_at_rank_idx(cand_top_tallies, election, voter_top_rank_idx)
    final_choices = election[
        np.arange(n_voters), voter_top_rank_idx
    ].copy()
    return CoombsResult(
        winner=winner,
        rounds=tuple(rounds),
        active_candidates=active_candidates,
        final_choices=final_choices,
        final_tallies=cand_top_tallies.copy(),
    )


def coombs_rounds(election, tiebreaker=None, *, stop_at=1):
    """
    Run Coombs' method and record each elimination round.

    Parameters
    ----------
    election : array_like
        A collection of complete ranked ballots. See `coombs` for the ballot
        format.
    tiebreaker : {'random', 'order', None}, optional
        Tie-breaking rule; see `coombs`.
    stop_at : int, optional
        Stop before eliminating below this number of active candidates.
        Majority winners still end the count immediately. The default of 1
        runs the ordinary Coombs count to a winner.

    Returns
    -------
    result : {CoombsResult, None}
        The count result and transfer trace, or ``None`` for an unbroken
        elimination tie.
    """
    return _run_coombs(
        election,
        tiebreaker=tiebreaker,
        stop_at=stop_at,
        record_rounds=True,
    )


def coombs(election, tiebreaker=None):
    """
    Find the winner of an election using Coomb's method.

    If any candidate gets a majority of first-preference votes, they win.
    Otherwise, the candidate(s) with the most number of last-preference votes
    is eliminated, votes for eliminated candidates are transferred according to
    the voters' preference rankings, and a series of runoff elections are held
    between the remainders until a candidate gets a majority. [1]_

    Parameters
    ----------
    election : array_like
        A collection of ranked ballots.  See `borda` for election format.
        Currently, this must include full rankings for each voter.
    tiebreaker : {'random', 'order', None}, optional
        If there is a tie, and `tiebreaker` is ``'random'``, tied candidates
        are eliminated or selected at random.
        If 'order', the lowest-ID tied candidate is preferred in each tie.
        By default, ``None`` is returned if there are any ties.

    Returns
    -------
    winner : {int, None}
        The ID number of the winner, or ``None`` for an unbroken tie.

    References
    ----------
    .. [1] :wikipedia:`Coombs' method`

    Examples
    --------
    Label some candidates:

    >>> A, B, C = 0, 1, 2

    Specify the ballots for the 5 voters:

    >>> election = [[A, C, B],
    ...             [A, C, B],
    ...             [B, C, A],
    ...             [B, C, A],
    ...             [C, A, B],
    ...             ]

    In the first round, no candidate gets a majority, so Candidate B (1) is
    eliminated, with 3 out of 5 last-place votes.  Voter 2 and 3's
    support of B is transferred to Candidate C (2), causing
    Candidate C to win, with 3 out of 5 votes:

    >>> coombs(election)
    2
    """
    result = _run_coombs(
        election,
        tiebreaker=tiebreaker,
        stop_at=1,
        record_rounds=False,
    )
    if result is None:
        return None
    if result.winner is not None:
        return result.winner
    raise RuntimeError("Bug in Coombs' calculation")
