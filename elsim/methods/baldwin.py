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

_tiebreak_map = {
    'order': _order_tiebreak_elim,
    'random': _random_tiebreak,
    None: _no_tiebreak,
}


@dataclass(frozen=True)
class BaldwinRound:
    """
    One Baldwin elimination and the resulting score changes.

    Attributes
    ----------
    eliminated : int
        Candidate with the lowest Borda score in this round.
    borda_before, borda_after : ndarray
        Borda scores immediately before and after the elimination. The
        project-wide convention gives the last active candidate one point.
    higher_ranked_candidates : tuple of ndarray
        For each voter, active candidates ranked above the eliminated
        candidate. Each loses one Borda point when the active field shrinks.
    first_tallies_before, first_tallies_after : ndarray
        First-choice tallies immediately before and after the elimination.
    transferred_voters : ndarray
        Indices of voters whose first choice was the eliminated candidate.
    transferred_to : ndarray
        New first-choice candidate for each voter in ``transferred_voters``.
    """

    eliminated: int
    borda_before: np.ndarray
    borda_after: np.ndarray
    higher_ranked_candidates: Tuple[np.ndarray, ...]
    first_tallies_before: np.ndarray
    first_tallies_after: np.ndarray
    transferred_voters: np.ndarray
    transferred_to: np.ndarray


@dataclass(frozen=True)
class BaldwinResult:
    """Result and round trace from Baldwin or Total Vote Runoff counting."""

    winner: Optional[int]
    rounds: Tuple[BaldwinRound, ...]
    active_candidates: np.ndarray
    final_choices: np.ndarray
    final_tallies: np.ndarray


def _borda_scores(election, eliminated_mask):
    """Return one-based Borda scores among the active candidates."""
    n_remaining = int(np.count_nonzero(~eliminated_mask))
    scores = np.zeros(election.shape[1], dtype=np.int64)
    for ballot in election:
        points = n_remaining
        for candidate in ballot:
            if eliminated_mask[candidate]:
                continue
            scores[candidate] += points
            points -= 1
    return scores


def _higher_ranked_candidates(election, eliminated_mask, eliminated):
    """Return each voter's active candidates above the round loser."""
    higher_per_voter = []
    for ballot in election:
        active_ballot = ballot[~eliminated_mask[ballot]]
        eliminated_rank = int(np.flatnonzero(
            active_ballot == eliminated
        )[0])
        higher_per_voter.append(
            active_ballot[:eliminated_rank].copy()
        )
    return tuple(higher_per_voter)


def _run_baldwin(
    election,
    tiebreaker,
    stop_at,
    stop_on_majority,
    record_rounds,
):
    """Run the shared count behind Baldwin and Total Vote Runoff."""
    election = np.asarray(election)
    n_voters, n_cands = election.shape
    stop_at = _validate_stop_at(stop_at, n_cands)
    tiebreak = _get_tiebreak(tiebreaker, _tiebreak_map)
    voter_top_rank_idx = np.zeros(n_voters, dtype=np.intp)
    cand_top_tallies = np.empty(n_cands, dtype=np.uint)
    eliminated_mask = np.zeros(n_cands, dtype=bool)
    rounds = []
    winner = None

    while np.count_nonzero(~eliminated_mask) > stop_at:
        _tally_at_rank_idx(
            cand_top_tallies, election, voter_top_rank_idx
        )
        cand_top_tallies_list = cand_top_tallies.tolist()

        if stop_on_majority:
            max_cand_top_tally = max(cand_top_tallies_list)
            if max_cand_top_tally > n_voters / 2:
                winner = cand_top_tallies_list.index(max_cand_top_tally)
                break

        borda_before = _borda_scores(election, eliminated_mask)
        active_scores = borda_before[~eliminated_mask]
        lowest_score = int(active_scores.min())
        low_scorers = [
            candidate
            for candidate in _all_indices(
                borda_before.tolist(), lowest_score
            )
            if not eliminated_mask[candidate]
        ]
        cand_to_eliminate = tiebreak(low_scorers)[0]
        if cand_to_eliminate is None:
            return None

        if record_rounds:
            choices_before = election[
                np.arange(n_voters), voter_top_rank_idx
            ].copy()
            first_tallies_before = cand_top_tallies.copy()
            higher_ranked_candidates = _higher_ranked_candidates(
                election, eliminated_mask, cand_to_eliminate
            )

        eliminated_mask[cand_to_eliminate] = True
        _inc_rank_idx(election, voter_top_rank_idx, eliminated_mask)

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
                BaldwinRound(
                    eliminated=int(cand_to_eliminate),
                    borda_before=borda_before,
                    borda_after=_borda_scores(
                        election, eliminated_mask
                    ),
                    higher_ranked_candidates=higher_ranked_candidates,
                    first_tallies_before=first_tallies_before,
                    first_tallies_after=cand_top_tallies.copy(),
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
    return BaldwinResult(
        winner=winner,
        rounds=tuple(rounds),
        active_candidates=active_candidates,
        final_choices=final_choices,
        final_tallies=cand_top_tallies.copy(),
    )


def baldwin_rounds(election, tiebreaker=None, *, stop_at=1):
    """
    Run Baldwin's method and record each Borda elimination.

    Baldwin's method repeatedly eliminates the candidate with the lowest Borda
    score, recalculating scores among the remaining candidates, until one
    candidate remains. Unlike Total Vote Runoff, it does not stop when a
    candidate obtains a majority of first choices.

    Parameters
    ----------
    election : array_like
        A collection of complete ranked ballots. See `borda` for the ballot
        format.
    tiebreaker : {'random', 'order', None}, optional
        If an elimination tie occurs, ``'random'`` chooses randomly,
        ``'order'`` eliminates the highest-ID tied candidate, and the default
        returns ``None``.
    stop_at : int, optional
        Stop before eliminating below this number of active candidates. The
        default of 1 runs the ordinary Baldwin count to a winner.

    Returns
    -------
    result : {BaldwinResult, None}
        The count result and score trace, or ``None`` for an unbroken
        elimination tie.

    References
    ----------
    .. [1] :wikipedia:`Nanson's method#Baldwin method`
    """
    return _run_baldwin(
        election,
        tiebreaker=tiebreaker,
        stop_at=stop_at,
        stop_on_majority=False,
        record_rounds=True,
    )


def total_vote_runoff_rounds(election, tiebreaker=None, *, stop_at=1):
    """
    Run Total Vote Runoff and record each Borda elimination.

    Total Vote Runoff uses Baldwin's lowest-Borda elimination rule but checks
    for a majority of first choices before each elimination, matching IRV's
    stopping rule. This operational distinction can shorten the trace even
    though both methods elect the same winner on complete strict rankings.

    Parameters
    ----------
    election : array_like
        A collection of complete ranked ballots. See `borda` for the ballot
        format.
    tiebreaker : {'random', 'order', None}, optional
        Tie-breaking rule; see `baldwin_rounds`.
    stop_at : int, optional
        Stop before eliminating below this number of active candidates.

    Returns
    -------
    result : {BaldwinResult, None}
        The count result and score trace, or ``None`` for an unbroken
        elimination tie.

    References
    ----------
    .. [1] Edward B. Foley, "Total Vote Runoff & Baldwin's method",
       Election Law Blog, 2022.
    """
    return _run_baldwin(
        election,
        tiebreaker=tiebreaker,
        stop_at=stop_at,
        stop_on_majority=True,
        record_rounds=True,
    )


def baldwin(election, tiebreaker=None):
    """
    Find the winner using Baldwin's iterative Borda elimination method.

    Parameters
    ----------
    election : array_like
        A collection of complete ranked ballots. See `borda` for the ballot
        format.
    tiebreaker : {'random', 'order', None}, optional
        Tie-breaking rule; see `baldwin_rounds`.

    Returns
    -------
    winner : {int, None}
        Candidate ID of the winner, or ``None`` for an unbroken tie.

    Examples
    --------
    >>> A, B, C = 0, 1, 2
    >>> election = [[A, C, B],
    ...             [A, C, B],
    ...             [B, C, A],
    ...             [B, C, A],
    ...             [C, A, B]]
    >>> baldwin(election)
    2
    """
    result = _run_baldwin(
        election,
        tiebreaker=tiebreaker,
        stop_at=1,
        stop_on_majority=False,
        record_rounds=False,
    )
    if result is None:
        return None
    if result.winner is not None:
        return result.winner
    raise RuntimeError("Bug in Baldwin's calculation")


def total_vote_runoff(election, tiebreaker=None):
    """
    Find the winner using Total Vote Runoff.

    Total Vote Runoff is operationally distinct from Baldwin's method because
    it declares a first-choice majority before calculating the next Borda
    elimination. The distinction changes the recorded rounds but not the
    winner for complete strict rankings.

    Parameters
    ----------
    election : array_like
        A collection of complete ranked ballots. See `borda` for the ballot
        format.
    tiebreaker : {'random', 'order', None}, optional
        Tie-breaking rule; see `baldwin_rounds`.

    Returns
    -------
    winner : {int, None}
        Candidate ID of the winner, or ``None`` for an unbroken tie.
    """
    result = _run_baldwin(
        election,
        tiebreaker=tiebreaker,
        stop_at=1,
        stop_on_majority=True,
        record_rounds=False,
    )
    if result is None:
        return None
    if result.winner is not None:
        return result.winner
    raise RuntimeError('Bug in Total Vote Runoff calculation')
