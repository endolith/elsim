from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from elsim.methods._common import (
    _all_indices,
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
class IRVRound:
    """
    One elimination and its resulting first-choice transfers.

    Attributes
    ----------
    eliminated : int
        Candidate removed in this round.
    tallies_before, tallies_after : ndarray
        First-choice tallies immediately before and after the elimination.
    transferred_voters : ndarray
        Indices of voters whose first choice was the eliminated candidate.
    transferred_to : ndarray
        New first-choice candidate for each voter in ``transferred_voters``.
    """

    eliminated: int
    tallies_before: np.ndarray
    tallies_after: np.ndarray
    transferred_voters: np.ndarray
    transferred_to: np.ndarray


@dataclass(frozen=True)
class IRVResult:
    """
    Result and round trace from an instant-runoff count.

    ``winner`` is ``None`` when counting stops with multiple candidates
    remaining. Candidates with no initial first-choice votes are excluded
    before the first elimination round and listed in
    ``initially_eliminated``.
    """

    winner: Optional[int]
    rounds: Tuple[IRVRound, ...]
    initially_eliminated: np.ndarray
    active_candidates: np.ndarray
    final_choices: np.ndarray
    final_tallies: np.ndarray


def _run_irv(election, tiebreaker, stop_at, record_rounds):
    """Run the shared IRV count used by the winner and trace APIs."""
    election = np.asarray(election)
    n_voters, n_cands = election.shape
    stop_at = _validate_stop_at(stop_at, n_cands)
    tiebreak = _get_tiebreak(tiebreaker, _tiebreak_map)
    voter_top_rank_idx = np.zeros(n_voters, dtype=np.intp)
    cand_tallies = np.empty(n_cands, dtype=np.uint)
    eliminated_mask = np.zeros(n_cands, dtype=bool)
    rounds = []
    winner = None

    # A candidate with no first choices cannot gain any transfers before
    # another candidate is eliminated. Excluding all such candidates together
    # preserves the historical IRV behavior without inventing an arbitrary
    # order among candidates tied at zero.
    _tally_at_rank_idx(cand_tallies, election, voter_top_rank_idx)
    initially_eliminated = np.flatnonzero(cand_tallies == 0)
    eliminated_mask[initially_eliminated] = True
    if initially_eliminated.size:
        _inc_rank_idx(election, voter_top_rank_idx, eliminated_mask)

    while np.count_nonzero(~eliminated_mask) > stop_at:
        _tally_at_rank_idx(cand_tallies, election, voter_top_rank_idx)
        cand_tallies_list = cand_tallies.tolist()

        max_cand_tally = max(cand_tallies_list)
        if max_cand_tally > n_voters / 2:
            winner = cand_tallies_list.index(max_cand_tally)
            break

        active_tallies = cand_tallies[~eliminated_mask]
        last_place_tally = int(active_tallies.min())
        last_place_cands = [
            candidate
            for candidate in _all_indices(cand_tallies_list, last_place_tally)
            if not eliminated_mask[candidate]
        ]
        cand_to_eliminate = tiebreak(last_place_cands)[0]
        if cand_to_eliminate is None:
            return None

        if record_rounds:
            choices_before = election[
                np.arange(n_voters), voter_top_rank_idx
            ].copy()
            tallies_before = cand_tallies.copy()

        eliminated_mask[cand_to_eliminate] = True
        _inc_rank_idx(election, voter_top_rank_idx, eliminated_mask)

        if record_rounds:
            choices_after = election[
                np.arange(n_voters), voter_top_rank_idx
            ].copy()
            _tally_at_rank_idx(
                cand_tallies, election, voter_top_rank_idx
            )
            transferred_voters = np.flatnonzero(
                choices_before == cand_to_eliminate
            )
            rounds.append(
                IRVRound(
                    eliminated=int(cand_to_eliminate),
                    tallies_before=tallies_before,
                    tallies_after=cand_tallies.copy(),
                    transferred_voters=transferred_voters,
                    transferred_to=choices_after[transferred_voters],
                )
            )

    active_candidates = np.flatnonzero(~eliminated_mask)
    if winner is None and active_candidates.size == 1:
        winner = int(active_candidates[0])

    _tally_at_rank_idx(cand_tallies, election, voter_top_rank_idx)
    final_choices = election[
        np.arange(n_voters), voter_top_rank_idx
    ].copy()
    return IRVResult(
        winner=winner,
        rounds=tuple(rounds),
        initially_eliminated=initially_eliminated,
        active_candidates=active_candidates,
        final_choices=final_choices,
        final_tallies=cand_tallies.copy(),
    )


def irv_rounds(election, tiebreaker=None, *, stop_at=1):
    """
    Run instant-runoff voting and record each elimination round.

    Parameters
    ----------
    election : array_like
        A collection of complete ranked ballots. See `irv` for the ballot
        format.
    tiebreaker : {'random', 'order', None}, optional
        Tie-breaking rule; see `irv`.
    stop_at : int, optional
        Stop before eliminating below this number of active candidates.
        Majority winners still end the count immediately. The default of 1
        runs the ordinary IRV count to a winner.

    Returns
    -------
    result : {IRVResult, None}
        The count result and transfer trace, or ``None`` for an unbroken
        elimination tie.

    Notes
    -----
    Candidates with no initial first-choice votes are excluded together before
    round recording begins. Their IDs are retained in
    ``result.initially_eliminated`` so a caller can reconstruct the complete
    count state.
    """
    return _run_irv(
        election,
        tiebreaker=tiebreaker,
        stop_at=stop_at,
        record_rounds=True,
    )


def irv(election, tiebreaker=None):
    """
    Find the winner of an election using instant-runoff voting.

    If any candidate gets a majority of first-preference votes, they win.
    Otherwise, the candidate(s) with the least number of first-choice votes
    is eliminated, votes for eliminated candidates are transferred according to
    the voters' preference rankings, and a series of runoff elections are held
    between the remainders until a candidate gets a majority. [1]_

    Also known as "the alternative vote", "ranked-choice voting", Hare's
    method, or Ware's method.

    The votes in each instant-runoff round are calculated from the same set of
    ranked ballots.  If voters are honest and consistent between rounds, then
    this is also equivalent to the exhaustive ballot method, which uses actual
    separate runoff elections. [2]_

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
    .. [1] :wikipedia:`Instant-runoff voting`
    .. [2] :wikipedia:`Exhaustive ballot`

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

    In the first round, no candidate gets a majority, so Candidate C (2) is
    eliminated, with 1 out of 5 first-place votes.  Voter 4's
    support of C is transferred to Candidate A (0), causing
    Candidate A to win, with 3 out of 5 votes:

    >>> irv(election)
    0
    """
    result = _run_irv(
        election,
        tiebreaker=tiebreaker,
        stop_at=1,
        record_rounds=False,
    )
    if result is None:
        return None
    if result.winner is not None:
        return result.winner
    raise RuntimeError('Bug in IRV calculation')
