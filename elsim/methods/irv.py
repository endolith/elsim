import numpy as np

from elsim.methods._common import (_all_indices, _get_tiebreak, _inc_rank_idx,
                                   _no_tiebreak, _order_tiebreak_elim,
                                   _random_tiebreak, _tally_at_rank_idx)

_tiebreak_map = {'order': _order_tiebreak_elim,
                 'random': _random_tiebreak,
                 None: _no_tiebreak}


def _irv_eliminate_until_n_remain(election, tiebreaker, n):
    """
    Same elimination rule as `irv`, but do not stop on a majority; keep
    eliminating the last-place active candidate until at most ``n`` remain.
    """
    n_voters, n_cands = election.shape
    tiebreak = _get_tiebreak(tiebreaker, _tiebreak_map)

    voter_top_rank_idx = np.zeros(n_voters, dtype=np.uint8)
    cand_tallies = np.empty(n_cands, dtype=np.uint)

    _tally_at_rank_idx(cand_tallies, election, voter_top_rank_idx)
    eliminated_cands = set(_all_indices(cand_tallies, 0))
    if eliminated_cands:
        _inc_rank_idx(election, voter_top_rank_idx, eliminated_cands)

    while n_cands - len(eliminated_cands) > n:
        _tally_at_rank_idx(cand_tallies, election, voter_top_rank_idx)
        cand_tallies_list = cand_tallies.tolist()
        active_tallies = [cand_tallies_list[c] for c in range(n_cands)
                          if c not in eliminated_cands]
        last_place_tally = min(active_tallies)
        last_place_cands = [c for c in range(n_cands)
                            if c not in eliminated_cands
                            and cand_tallies_list[c] == last_place_tally]
        cand_to_eliminate = tiebreak(last_place_cands)[0]
        if cand_to_eliminate is None:
            return None
        eliminated_cands.add(cand_to_eliminate)
        _inc_rank_idx(election, voter_top_rank_idx, eliminated_cands)

    return {c for c in range(n_cands) if c not in eliminated_cands}


def irv(election, tiebreaker=None, *, n_survivors=None):
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
    n_survivors : int, optional
        If omitted (default), return the usual single IRV winner.
        If ``1``, same as omitted (one winner).
        If ``k`` with ``1 < k < n_candidates``, return the set of candidate IDs
        still standing after repeatedly eliminating the last-place active
        candidate (same transfer rule as above) until ``k`` remain, without
        stopping early when someone has a majority.  This models a ranked
        primary that narrows the field to ``k`` for a later round.
        If ``k`` is greater than or equal to the number of candidates, every
        candidate is returned as a set.

    Returns
    -------
    outcome : {int, set of int, None}
        If ``n_survivors`` is omitted or ``1``, the winner's ID, or ``None`` for
        an unbroken tie.
        If ``n_survivors`` is strictly between ``1`` and the number of
        candidates, the set of surviving candidate IDs, or ``None`` for an
        unbroken tie during elimination.
        If ``n_survivors`` is greater than or equal to the number of candidates,
        the set of all candidate IDs.

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

    The same ballots with ``n_survivors`` = 2 (no majority short-circuit;
    eliminate last place once so two candidates remain):

    >>> election_b = [[A, C, B],
    ...               [A, C, B],
    ...               [B, C, A],
    ...               [B, C, A],
    ...               [C, A, B],
    ...               ]
    >>> sorted(irv(election_b, n_survivors=2, tiebreaker='order'))
    [0, 1]
    """
    election = np.asarray(election)
    n_voters, n_cands = election.shape

    if n_survivors is not None:
        if n_survivors < 1:
            raise ValueError('n_survivors must be at least 1')
        if n_survivors >= n_cands:
            return set(range(n_cands))
        if n_survivors > 1:
            return _irv_eliminate_until_n_remain(election, tiebreaker,
                                                 n_survivors)

    tiebreak = _get_tiebreak(tiebreaker, _tiebreak_map)
    voter_top_rank_idx = np.zeros(n_voters, dtype=np.uint8)
    cand_tallies = np.empty(n_cands, dtype=np.uint)

    # Eliminate candidates with no first-choice votes before rounds begin.
    # TODO: In the future when round tallies are also output, this should be
    # its own round.  Either eliminate one zero-voted candidate at a time, or
    # do a batch elimination of all candidates who can't possibly win in each
    # round.  (Probably have a batch_elimination=True flag to choose.)
    # Currently this step is needed because eliminated candidates drop to zero
    # votes and can't be distinguished from candidates who never received any.
    _tally_at_rank_idx(cand_tallies, election, voter_top_rank_idx)
    eliminated_mask = np.zeros(n_cands, dtype=bool)
    eliminated_mask[_all_indices(cand_tallies, 0)] = True
    _inc_rank_idx(election, voter_top_rank_idx, eliminated_mask)

    for round_ in range(n_cands):
        _tally_at_rank_idx(cand_tallies, election, voter_top_rank_idx)

        # (tolist makes things 2-4x faster)
        cand_tallies_list = cand_tallies.tolist()

        # Did anyone get a majority?
        max_cand_tally = max(cand_tallies_list)
        if max_cand_tally > n_voters / 2:
            return cand_tallies_list.index(max_cand_tally)

        # If not, eliminate least-favorited candidate
        # (generator is faster than min(arr[np.nonzero(arr)]) for small lists)
        last_place_tally = min(t for t in cand_tallies_list if t != 0)
        last_place_cands = _all_indices(cand_tallies_list, last_place_tally)
        cand_to_eliminate = tiebreak(last_place_cands)[0]

        if cand_to_eliminate is None:
            # No tiebreaker case
            return None
        else:
            eliminated_mask[cand_to_eliminate] = True

        # Increment rank indices past all eliminated candidates
        _inc_rank_idx(election, voter_top_rank_idx, eliminated_mask)

    raise RuntimeError('Bug in IRV calculation')
