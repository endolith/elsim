import numpy as np

from elsim.methods._common import (_all_indices, _get_tiebreak, _inc_rank_idx,
                                   _no_tiebreak, _order_tiebreak_elim,
                                   _random_tiebreak, _tally_at_rank_idx)

_tiebreak_map = {'order': _order_tiebreak_elim,
                 'random': _random_tiebreak,
                 None: _no_tiebreak}


def _compute_borda_scores(election, eliminated_mask):
    """
    Borda scores for non-eliminated candidates as if eliminated were never on the ballot.

    Scoring among remaining candidates: ``n_remaining - 1`` points for 1st
    place, 0 for last place.  Eliminated candidates receive score 0.

    Parameters
    ----------
    election : ndarray
        Ranked ballots (n_voters × n_cands).
    eliminated_mask : ndarray of bool, length n_cands
        ``eliminated_mask[i]`` is True if candidate ``i`` is eliminated.

    Returns
    -------
    scores : ndarray of float, length n_cands
        Borda scores; eliminated candidates have score 0.
    """
    n_remaining = int(np.sum(~eliminated_mask))
    scores = np.zeros(election.shape[1], dtype=float)
    for ballot in election:
        pos = 0
        for cand_id in ballot:
            if eliminated_mask[cand_id]:
                continue
            scores[cand_id] += (n_remaining - 1 - pos)
            pos += 1
    return scores


def baldwin_rounds(election, tiebreaker=None, *, min_remaining=1, record_rounds=False):
    """
    Run Baldwin's method (Total Vote Runoff) and return per-round elimination data.

    Baldwin's method re-tallies Borda scores each round among remaining
    candidates only and eliminates the lowest scorer.  A candidate with a
    majority of first-preference votes wins immediately.

    Parameters
    ----------
    election : array_like
        Ranked ballots; see `borda` for election format.
    tiebreaker : {'random', 'order', None}, optional
        If there is an elimination tie, ``'random'`` picks at random,
        ``'order'`` eliminates the highest-ID tied candidate, and ``None``
        (default) returns ``None`` for the whole result.
    min_remaining : int, optional
        Stop eliminating when this many candidates remain (default 1).
        A majority of first-preference votes ends the count early regardless.
    record_rounds : bool, optional
        If True (default False), each round entry in ``rounds`` includes
        Borda snapshots and promoted-voter data for animation.

    Returns
    -------
    result : dict or None
        ``None`` if there is an elimination tie and ``tiebreaker`` is ``None``.
        Otherwise a dict with:

        winner : int or None
            Candidate ID of the majority winner, or the sole remaining
            candidate when exactly one is left, or ``None``.
        rounds : list of dict
            One entry per elimination round.  With ``record_rounds=True``,
            each dict has ``loser``, ``borda_before``, ``borda_after``,
            ``promoted_per_voter``.  Without ``record_rounds``, each dict
            has only ``loser``.
        eliminated_mask : ndarray of bool
            Final elimination state.
        final_ballots : ndarray
            Per-voter top remaining choice at end (length n_voters).
        final_tallies : ndarray
            First-preference tallies at end (length n_cands).
    """
    election = np.asarray(election)
    n_voters, n_cands = election.shape
    tiebreak = _get_tiebreak(tiebreaker, _tiebreak_map)
    voter_top_rank_idx = np.zeros(n_voters, dtype=np.uint8)
    cand_top_tallies = np.empty(n_cands, dtype=np.uint)
    eliminated_mask = np.zeros(n_cands, dtype=bool)
    rounds = []
    winner = None

    while np.sum(~eliminated_mask) > min_remaining:
        _tally_at_rank_idx(cand_top_tallies, election, voter_top_rank_idx)
        cand_top_tallies_list = cand_top_tallies.tolist()

        # Did anyone get a majority of first-preference votes?
        max_cand_top_tally = max(cand_top_tallies_list)
        if max_cand_top_tally > n_voters / 2:
            winner = cand_top_tallies_list.index(max_cand_top_tally)
            break

        # If not, eliminate the candidate with the lowest Borda score
        borda_before = _compute_borda_scores(election, eliminated_mask)
        borda_list = borda_before.tolist()
        min_score = min(borda_list[c] for c in range(n_cands) if not eliminated_mask[c])
        low_scorers = [c for c in _all_indices(borda_list, min_score) if not eliminated_mask[c]]
        cand_to_eliminate = tiebreak(low_scorers)[0]

        if cand_to_eliminate is None:
            return None

        if record_rounds:
            # For each voter: the active candidates ranked below the loser.
            # When the loser is removed, each of these gains +1 Borda point.
            promoted_per_voter = []
            for ballot in election:
                promoted = []
                found_loser = False
                for cand_id in ballot:
                    if eliminated_mask[cand_id]:
                        continue
                    if cand_id == cand_to_eliminate:
                        found_loser = True
                        continue
                    if found_loser:
                        promoted.append(cand_id)
                promoted_per_voter.append(promoted)

        eliminated_mask[cand_to_eliminate] = True
        _inc_rank_idx(election, voter_top_rank_idx, eliminated_mask)

        if record_rounds:
            borda_after = _compute_borda_scores(election, eliminated_mask)
            rounds.append({
                'loser': cand_to_eliminate,
                'borda_before': borda_before,
                'borda_after': borda_after,
                'promoted_per_voter': promoted_per_voter,
            })
        else:
            rounds.append({'loser': cand_to_eliminate})

    if winner is None:
        remaining = np.flatnonzero(~eliminated_mask)
        if len(remaining) == 1:
            winner = int(remaining[0])

    _tally_at_rank_idx(cand_top_tallies, election, voter_top_rank_idx)
    final_ballots = election[np.arange(n_voters), voter_top_rank_idx].copy()

    return {
        'winner': winner,
        'rounds': rounds,
        'eliminated_mask': eliminated_mask,
        'final_ballots': final_ballots,
        'final_tallies': cand_top_tallies.copy(),
    }


def baldwin(election, tiebreaker=None):
    """
    Find the winner of an election using Baldwin's method (Total Vote Runoff).

    Borda scores are re-tallied each round among remaining candidates only,
    and the candidate with the lowest score is eliminated.  If any candidate
    has a majority of first-preference votes at any point, they win
    immediately. [1]_

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
    .. [1] :wikipedia:`Nanson's method#Baldwin method`

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

    B has the lowest Borda score and is eliminated.  C then wins with a
    majority.  (Under IRV, C is eliminated first and A wins — a different
    result.)

    >>> baldwin(election)
    2
    """
    result = baldwin_rounds(election, tiebreaker=tiebreaker)
    if result is None:
        return None
    winner = result['winner']
    if winner is not None:
        return winner
    raise RuntimeError("Bug in Baldwin's calculation")
