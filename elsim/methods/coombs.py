import numpy as np

from elsim.methods._common import (_all_indices, _dec_rank_idx, _get_tiebreak,
                                   _inc_rank_idx, _no_tiebreak,
                                   _order_tiebreak_elim, _random_tiebreak,
                                   _tally_at_rank_idx)

_tiebreak_map = {'order': _order_tiebreak_elim,
                 'random': _random_tiebreak,
                 None: _no_tiebreak}


def coombs_rounds(election, tiebreaker=None, *, min_remaining=1, record_rounds=False):
    """
    Run Coombs' method and return per-round elimination data.

    Parameters
    ----------
    election : array_like
        Ranked ballots; see `coombs`.
    tiebreaker : {'random', 'order', None}, optional
        Tie-breaking rule; see `coombs`.
    min_remaining : int, optional
        Stop eliminating when this many candidates remain (default 1).
        A majority of first-preference votes ends the count early regardless.
    record_rounds : bool, optional
        If True (default False), each round entry in ``rounds`` includes
        ballot snapshots and affected-voter indices for animation.

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
            each dict has ``loser``, ``ballots_before``, ``ballots_after``,
            ``tallies_before``, ``affected_voters``.  Without
            ``record_rounds``, each dict has only ``loser``.
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
    voter_bottom_rank_idx = np.full(n_voters, n_cands - 1, dtype=np.uint8)
    cand_bottom_tallies = np.empty(n_cands, dtype=np.uint)
    eliminated_mask = np.zeros(n_cands, dtype=bool)
    rounds = []
    winner = None

    # No IRV-style eager elimination of zero top tallies here: Coombs'
    # elimination criterion is last-place tallies, not lowest top tallies.

    while np.sum(~eliminated_mask) > min_remaining:
        _tally_at_rank_idx(cand_top_tallies, election, voter_top_rank_idx)

        # (tolist makes things 2-4x faster)
        cand_top_tallies_list = cand_top_tallies.tolist()

        # Did anyone get a majority?
        max_cand_top_tally = max(cand_top_tallies_list)
        if max_cand_top_tally > n_voters / 2:
            winner = cand_top_tallies_list.index(max_cand_top_tally)
            break

        # If not, eliminate candidate with the most last-place votes
        _tally_at_rank_idx(cand_bottom_tallies, election, voter_bottom_rank_idx)
        # (tolist makes things 2-4x faster)
        cand_bottom_tallies_list = cand_bottom_tallies.tolist()
        max_cand_bottom_tally = max(cand_bottom_tallies_list)
        max_bottom_tally_cands = _all_indices(cand_bottom_tallies_list,
                                              max_cand_bottom_tally)
        cand_to_eliminate = tiebreak(max_bottom_tally_cands)[0]

        if cand_to_eliminate is None:
            return None

        if record_rounds:
            ballots_before = election[np.arange(n_voters), voter_top_rank_idx].copy()
            affected_voters = np.flatnonzero(ballots_before == cand_to_eliminate)

        eliminated_mask[cand_to_eliminate] = True

        # Increment/decrement rank indices past all eliminated candidates
        # (top and bottom rank indices move in opposite directions)
        _inc_rank_idx(election, voter_top_rank_idx, eliminated_mask)
        _dec_rank_idx(election, voter_bottom_rank_idx, eliminated_mask)

        if record_rounds:
            ballots_after = election[np.arange(n_voters), voter_top_rank_idx].copy()
            rounds.append({
                'loser': cand_to_eliminate,
                'ballots_before': ballots_before,
                'ballots_after': ballots_after,
                'tallies_before': cand_top_tallies.copy(),
                'affected_voters': affected_voters,
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
    result = coombs_rounds(election, tiebreaker=tiebreaker)
    if result is None:
        return None
    winner = result['winner']
    if winner is not None:
        return winner
    raise RuntimeError("Bug in Coombs' calculation")
