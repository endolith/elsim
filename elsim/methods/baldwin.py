import numpy as np

from elsim.methods._common import (
    _all_indices,
    _get_tiebreak,
    _inc_rank_idx,
    _no_tiebreak,
    _order_tiebreak_elim,
    _random_tiebreak,
    _tally_at_rank_idx,
)

_tiebreak_map = {
    'order': _order_tiebreak_elim,
    'random': _random_tiebreak,
    None: _no_tiebreak,
}


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


def _run_baldwin(election, tiebreaker):
    """Run Baldwin's lowest-Borda count with its majority stopping rule.

    Repeatedly eliminate the active candidate with the lowest Borda score,
    recomputing scores among the remaining candidates, until one candidate
    remains or a candidate holds a first-choice majority. A majority
    candidate is the Condorcet winner and necessarily wins.
    """
    election = np.asarray(election)
    n_voters, n_cands = election.shape
    tiebreak = _get_tiebreak(tiebreaker, _tiebreak_map)
    voter_top_rank_idx = np.zeros(n_voters, dtype=np.intp)
    cand_top_tallies = np.empty(n_cands, dtype=np.uint)
    eliminated_mask = np.zeros(n_cands, dtype=bool)

    while np.count_nonzero(~eliminated_mask) > 1:
        _tally_at_rank_idx(cand_top_tallies, election, voter_top_rank_idx)
        cand_top_tallies_list = cand_top_tallies.tolist()

        max_cand_top_tally = max(cand_top_tallies_list)
        if max_cand_top_tally > n_voters / 2:
            return cand_top_tallies_list.index(max_cand_top_tally)

        borda_scores = _borda_scores(election, eliminated_mask)
        active_scores = borda_scores[~eliminated_mask]
        lowest_score = int(active_scores.min())
        low_scorers = [
            candidate
            for candidate in _all_indices(borda_scores.tolist(), lowest_score)
            if not eliminated_mask[candidate]
        ]
        cand_to_eliminate = tiebreak(low_scorers)[0]
        if cand_to_eliminate is None:
            return None

        eliminated_mask[cand_to_eliminate] = True
        _inc_rank_idx(election, voter_top_rank_idx, eliminated_mask)

    return int(np.flatnonzero(~eliminated_mask)[0])


def baldwin(election, tiebreaker=None):
    """
    Find the winner using Baldwin's iterative Borda elimination method.

    Baldwin repeatedly eliminates the lowest-Borda candidate, recomputing
    scores among the remaining candidates, until one remains or a candidate
    obtains a first-choice majority. Because a Condorcet winner always has an
    above-average Borda score, it can never be the lowest-scoring candidate
    and is never eliminated, so Baldwin satisfies the Condorcet criterion.

    Parameters
    ----------
    election : array_like
        A collection of complete ranked ballots. See `borda` for the ballot
        format.
    tiebreaker : {'random', 'order', None}, optional
        If an elimination tie occurs, ``'random'`` chooses randomly,
        ``'order'`` eliminates the highest-ID tied candidate, and the default
        of ``None`` returns ``None``.

    Returns
    -------
    winner : {int, None}
        Candidate ID of the winner, or ``None`` for an unbroken tie.

    References
    ----------
    .. [1] :wikipedia:`Nanson's method#Baldwin method`

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
    return _run_baldwin(election, tiebreaker)


def total_vote_runoff(election, tiebreaker=None):
    """
    Find the winner using Total Vote Runoff.

    Total Vote Runoff, the name used by Foley and Maskin (2022), is the same
    lowest-Borda count as Baldwin's method, including its first-choice-
    majority stopping rule, and therefore also satisfies the Condorcet
    criterion.

    Parameters
    ----------
    election : array_like
        A collection of complete ranked ballots. See `borda` for the ballot
        format.
    tiebreaker : {'random', 'order', None}, optional
        Tie-breaking rule; see `baldwin`.

    Returns
    -------
    winner : {int, None}
        Candidate ID of the winner, or ``None`` for an unbroken tie.

    References
    ----------
    .. [1] Edward B. Foley, "Total Vote Runoff & Baldwin's method",
       Election Law Blog, 2022.
    """
    return baldwin(election, tiebreaker=tiebreaker)
