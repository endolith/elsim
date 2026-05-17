import numpy as np

from elsim.methods._common import (_all_indices, _get_tiebreak, _inc_rank_idx,
                                   _no_tiebreak, _order_tiebreak_elim,
                                   _random_tiebreak, _tally_at_rank_idx)

_tiebreak_map = {'order': _order_tiebreak_elim,
                 'random': _random_tiebreak,
                 None: _no_tiebreak}


def irv_rounds(election, tiebreaker=None, *, min_remaining=1, record_trace=False):
    """
    Run instant-runoff voting and optionally record per-round elimination data.

    Parameters
    ----------
    election : array_like
        Ranked ballots; see `irv`.
    tiebreaker : {'random', 'order', None}, optional
        Tie-breaking rule when eliminating last-place candidates; see `irv`.
    min_remaining : int, optional
        Stop eliminating when this many candidates remain (default 1).
        A majority winner ends the count early regardless.
    record_trace : bool, optional
        If True, each elimination round includes ballot and tally snapshots
        for animation (see Returns).

    Returns
    -------
    result : dict or None
        ``None`` if elimination is tied and ``tiebreaker`` is ``None``.
        Otherwise a dict with:

        winner : int or None
            Majority winner if one was found before stopping; else the sole
            remaining candidate when exactly one is left; else ``None``.
        rounds : list of dict
            One entry per elimination round (empty if none).  With
            ``record_trace=True``, each dict has keys ``loser``,
            ``ballots_before``, ``ballots_after``, ``tallies_before``,
            ``affected_voters``.
        eliminated_mask : ndarray of bool
            Final elimination state.
        final_ballots : ndarray
            Per-voter top remaining choice at the end (length ``n_voters``).
        final_tallies : ndarray
            First-preference tallies at the end (length ``n_cands``).
    """
    election = np.asarray(election)
    n_voters, n_cands = election.shape
    tiebreak = _get_tiebreak(tiebreaker, _tiebreak_map)
    voter_top_rank_idx = np.zeros(n_voters, dtype=np.uint8)
    cand_tallies = np.empty(n_cands, dtype=np.uint)
    eliminated_mask = np.zeros(n_cands, dtype=bool)
    rounds = []
    winner = None

    # Eliminate candidates with no first-choice votes before rounds begin.
    # TODO: In the future when round tallies are also output, this should be
    # its own round.  Either eliminate one zero-voted candidate at a time, or
    # do a batch elimination of all candidates who can't possibly win in each
    # round.  (Probably have a batch_elimination=True flag to choose.)
    # Currently this step is needed because eliminated candidates drop to zero
    # votes and can't be distinguished from candidates who never received any.
    _tally_at_rank_idx(cand_tallies, election, voter_top_rank_idx)
    for cand in _all_indices(cand_tallies.tolist(), 0):
        eliminated_mask[cand] = True
    if np.any(eliminated_mask):
        _inc_rank_idx(election, voter_top_rank_idx, eliminated_mask)

    while np.sum(~eliminated_mask) > min_remaining:
        _tally_at_rank_idx(cand_tallies, election, voter_top_rank_idx)

        # (tolist makes things 2-4x faster)
        cand_tallies_list = cand_tallies.tolist()

        # Did anyone get a majority?
        max_cand_tally = max(cand_tallies_list)
        if max_cand_tally > n_voters / 2:
            winner = cand_tallies_list.index(max_cand_tally)
            break

        # If not, eliminate least-favorited candidate
        # (generator is faster than min(arr[np.nonzero(arr)]) for small lists)
        last_place_tally = min(t for t in cand_tallies_list if t != 0)
        last_place_cands = _all_indices(cand_tallies_list, last_place_tally)
        cand_to_eliminate = tiebreak(last_place_cands)[0]

        if cand_to_eliminate is None:
            return None

        if record_trace:
            ballots_before = election[np.arange(n_voters), voter_top_rank_idx].copy()
            affected_voters = np.flatnonzero(ballots_before == cand_to_eliminate)

        eliminated_mask[cand_to_eliminate] = True
        _inc_rank_idx(election, voter_top_rank_idx, eliminated_mask)

        if record_trace:
            ballots_after = election[np.arange(n_voters), voter_top_rank_idx].copy()
            rounds.append({
                'loser': cand_to_eliminate,
                'ballots_before': ballots_before,
                'ballots_after': ballots_after,
                'tallies_before': cand_tallies.copy(),
                'affected_voters': affected_voters,
            })

    if winner is None:
        remaining = np.flatnonzero(~eliminated_mask)
        if len(remaining) == 1:
            winner = int(remaining[0])

    _tally_at_rank_idx(cand_tallies, election, voter_top_rank_idx)
    final_ballots = election[np.arange(n_voters), voter_top_rank_idx].copy()

    return {
        'winner': winner,
        'rounds': rounds,
        'eliminated_mask': eliminated_mask,
        'final_ballots': final_ballots,
        'final_tallies': cand_tallies.copy(),
    }


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
    result = irv_rounds(election, tiebreaker=tiebreaker, min_remaining=1,
                        record_trace=False)
    if result is None:
        return None
    winner = result['winner']
    if winner is not None:
        return winner
    raise RuntimeError('Bug in IRV calculation')
