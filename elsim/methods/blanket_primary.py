"""
Two-round systems with a primary that narrows the field (blanket / top-n).

These model nonpartisan blanket primaries and related reforms: approval plus
runoff (unified primary), pick-one top-four or top-five with a ranked general
(Final Four / Final Five), IRV-style primaries that leave a fixed slate, and
Condorcet in the general per :wikipedia:`Top-four primary` variations.
"""
import numpy as np

from elsim.methods._common import (_all_indices, _get_tiebreak, _inc_rank_idx,
                                   _no_tiebreak, _order_tiebreak_elim,
                                   _order_tiebreak_keep, _random_tiebreak,
                                   _tally_at_rank_idx)
from elsim.methods.condorcet import condorcet
from elsim.methods.fptp import sntv
from elsim.methods.irv import irv
from elsim.methods.runoff import runoff

_tiebreak_map_keep = {'order': _order_tiebreak_keep,
                      'random': _random_tiebreak,
                      None: _no_tiebreak}

_tiebreak_map_elim = {'order': _order_tiebreak_elim,
                      'random': _random_tiebreak,
                      None: _no_tiebreak}


def _top_n_from_plurality_tallies(tallies, n, tiebreaker):
    """
    Candidate indices with the top ``n`` first-preference counts, breaking
    boundary ties the same way as ``sntv``.
    """
    tallies = np.asarray(tallies)
    n_cands = len(tallies)
    if n < 1:
        raise ValueError('n must be at least 1')
    if n >= n_cands:
        return set(range(n_cands))

    sorted_indices = np.argsort(tallies)
    top_candidates = sorted_indices[-n:]
    worst_top = top_candidates[0]
    best_not_top = sorted_indices[-n - 1]

    if tallies[worst_top] == tallies[best_not_top]:
        untied_winners = sorted_indices[tallies[sorted_indices] >
                                        tallies[worst_top]]
        tied_candidates = np.where(tallies == tallies[worst_top])[0]
        tiebreak = _get_tiebreak(tiebreaker, _tiebreak_map_keep)
        n_needed = n - len(untied_winners)
        tie_winners = tiebreak(list(tied_candidates), n_needed)
        if tie_winners == [None]:
            return None
        winners = set(untied_winners) | set(tie_winners)
    else:
        winners = set(top_candidates)

    return set(int(w) for w in winners)


def _primary_top_n_approval(approval_election, n, tiebreaker):
    approval_election = np.asarray(approval_election, dtype=np.uint8)
    if approval_election.min() < 0 or approval_election.max() > 1:
        raise ValueError('Approval ballots must contain only 0 and 1')
    tallies = approval_election.sum(axis=0)
    n_cands = approval_election.shape[1]
    if n >= n_cands:
        return set(range(n_cands))
    return _top_n_from_plurality_tallies(tallies, n, tiebreaker)


def _restrict_ballots(election, allowed):
    """Drop non-finalists from each ballot, preserving order (local IDs 0..k-1)."""
    election = np.asarray(election)
    allowed_sorted = sorted(allowed)
    old_to_new = {cand: j for j, cand in enumerate(allowed_sorted)}
    new_to_old = {j: cand for j, cand in enumerate(allowed_sorted)}
    rows = []
    for ballot in election:
        row = [old_to_new[c] for c in ballot.tolist() if c in old_to_new]
        rows.append(row)
    return np.asarray(rows, dtype=election.dtype), new_to_old


def _head_to_head_two(finalist_0, finalist_1, election, tiebreaker):
    """Pairwise winner between two finalists using full rankings (contingent vote)."""
    tiebreak = _get_tiebreak(tiebreaker, _tiebreak_map_keep)
    n_voters = election.shape[0]
    finalist_0_tally = 0
    finalist_1_tally = 0
    for ballot in election:
        ballot_list = ballot.tolist()
        if ballot_list.index(finalist_0) < ballot_list.index(finalist_1):
            finalist_0_tally += 1
        else:
            finalist_1_tally += 1
    assert finalist_0_tally + finalist_1_tally == n_voters
    if finalist_0_tally == finalist_1_tally:
        return tiebreak([finalist_0, finalist_1])[0]
    if finalist_0_tally > finalist_1_tally:
        return finalist_0
    return finalist_1


def irv_eliminate_to_n(election, n, tiebreaker=None):
    """
    Sequential-elimination primary: repeatedly eliminate the IRV last-place
    candidate among remaining candidates until ``n`` (or fewer) remain.

    This is the ranked primary used in some top-four reform variants. [1]_

    Parameters
    ----------
    election : array_like
        Ranked ballots (same format as ``irv``).
    n : int
        Number of candidates that should remain.
    tiebreaker : {'random', 'order', None}, optional
        Same meaning as in ``irv`` (who to eliminate when last-place is tied).

    Returns
    -------
    finalists : {set of int, None}
        Candidate IDs still in the race, or ``None`` if a tie could not be
        broken while eliminating.

    References
    ----------
    .. [1] :wikipedia:`Top-four primary#Variations`

    Examples
    --------
    >>> election = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
    >>> sorted(irv_eliminate_to_n(election, 2, tiebreaker='order'))
    [0, 1]
    """
    election = np.asarray(election)
    n_voters, n_cands = election.shape
    if n < 1:
        raise ValueError('n must be at least 1')
    if n >= n_cands:
        return set(range(n_cands))

    tiebreak = _get_tiebreak(tiebreaker, _tiebreak_map_elim)

    voter_top_rank_idx = np.zeros(n_voters, dtype=np.uint8)
    cand_tallies = np.empty(n_cands, dtype=np.uint)

    _tally_at_rank_idx(cand_tallies, election, voter_top_rank_idx)
    eliminated_cands = set(_all_indices(cand_tallies.tolist(), 0))
    if eliminated_cands:
        _inc_rank_idx(election, voter_top_rank_idx, eliminated_cands)

    while n_cands - len(eliminated_cands) > n:
        _tally_at_rank_idx(cand_tallies, election, voter_top_rank_idx)
        if n_cands - len(eliminated_cands) <= n:
            break
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


def approval_runoff(approval_election, ranked_election, tiebreaker=None):
    """
    Unified primary / top-two approval plus runoff (St. Louis style). [1]_

    The two candidates with the most approvals advance; the winner is decided
    by pairwise majority on the ranked ballots (contingent vote between the
    two finalists), the usual ranked-ballot model when the general uses a
    separate runoff.

    Parameters
    ----------
    approval_election : array_like
        Approval ballots: rows are voters, columns candidates, entries 0/1.
    ranked_election : array_like
        Complete ranked ballots for the same voters (same number of rows as
        ``approval_election``).
    tiebreaker : {'random', 'order', None}, optional
        Used for selecting primary finalists and for a tied general.

    Returns
    -------
    winner : {int, None}

    References
    ----------
    .. [1] :wikipedia:`Unified primary`

    Examples
    --------
    >>> A, B, C = 0, 1, 2
    >>> approvals = [[1, 1, 0], [1, 1, 0], [0, 1, 1]]
    >>> ranked = [[B, A, C], [B, C, A], [C, B, A]]
    >>> approval_runoff(approvals, ranked)
    1
    """
    approval_election = np.asarray(approval_election, dtype=np.uint8)
    ranked_election = np.asarray(ranked_election)
    if approval_election.shape[0] != ranked_election.shape[0]:
        raise ValueError('approval_election and ranked_election must have the '
                         'same number of rows (voters)')
    if approval_election.min() < 0 or approval_election.max() > 1:
        raise ValueError('Approval ballots must contain only 0 and 1')

    finalists = _primary_top_n_approval(approval_election, 2, tiebreaker)
    if finalists is None:
        return None
    if len(finalists) == 1:
        (only,) = tuple(finalists)
        return only
    finalist_0, finalist_1 = sorted(finalists)
    out = _head_to_head_two(finalist_0, finalist_1, ranked_election,
                            tiebreaker)
    return int(out) if out is not None else None


def top_n_irv(election, n, tiebreaker=None):
    """
    Pick-one top-``n`` primary (plurality / SNTV) and an IRV general. [1]_

    The ``n`` candidates with the most first-preference votes advance; the
    winner is chosen by instant-runoff on the same rankings restricted to
    those finalists (Alaska-style Final Four when ``n`` is 4).

    Parameters
    ----------
    election : array_like
        Ranked ballots.
    n : int
        Primary cutoff (4 for Final Four, 5 for Final Five).
    tiebreaker : {'random', 'order', None}, optional

    Returns
    -------
    winner : {int, None}

    References
    ----------
    .. [1] :wikipedia:`Top-four primary`

    Examples
    --------
    >>> A, B, C = 0, 1, 2
    >>> election = [*6*[[A, B, C]], *3*[[B, A, C]], *1*[[C, B, A]]]
    >>> top_n_irv(election, 2)
    0
    """
    finalists = sntv(election, n, tiebreaker)
    if finalists is None:
        return None
    sub, new_to_old = _restrict_ballots(election, finalists)
    w = irv(sub, tiebreaker)
    if w is None:
        return None
    return int(new_to_old[w])


def top_n_runoff(election, n, tiebreaker=None):
    """
    Pick-one top-``n`` primary and a top-two contingent general. [1]_

    After the top ``n`` candidates by plurality advance, the general election
    applies the same two-candidate contingent logic as ``runoff`` on the
    restricted set (first round among finalists only, then pairwise between the
    top two by first preference among finalists).

    Parameters
    ----------
    election : array_like
        Ranked ballots.
    n : int
        Primary cutoff.
    tiebreaker : {'random', 'order', None}, optional

    Returns
    -------
    winner : {int, None}

    References
    ----------
    .. [1] :wikipedia:`Top-four primary#Variations`

    Examples
    --------
    >>> A, B, C, D = 0, 1, 2, 3
    >>> election = [*3*[[A, B, C, D]], *2*[[B, A, C, D]], *1*[[C, D, A, B]]]
    >>> isinstance(top_n_runoff(election, 4), int)
    True
    """
    finalists = sntv(election, n, tiebreaker)
    if finalists is None:
        return None
    sub, new_to_old = _restrict_ballots(election, finalists)
    w = runoff(sub, tiebreaker)
    if w is None:
        return None
    return int(new_to_old[w])


def top_n_condorcet(election, n, tiebreaker=None):
    """
    Pick-one top-``n`` primary and a Condorcet pairwise general. [1]_

    Parameters
    ----------
    election : array_like
        Ranked ballots.
    n : int
        Primary cutoff.
    tiebreaker : {'random', 'order', None}, optional
        Used only for the primary; the general has no tiebreaker (same as
        ``condorcet``).

    Returns
    -------
    winner : {int, None}

    References
    ----------
    .. [1] :wikipedia:`Top-four primary#Variations`

    Examples
    --------
    >>> A, B, C = 0, 1, 2
    >>> election = [[A, B, C], [A, B, C], [B, A, C]]
    >>> top_n_condorcet(election, 2)
    0
    """
    finalists = sntv(election, n, tiebreaker)
    if finalists is None:
        return None
    sub, new_to_old = _restrict_ballots(election, finalists)
    w = condorcet(sub)
    if w is None:
        return None
    return int(new_to_old[w])


def irv_primary_top_n_irv(election, n, tiebreaker=None):
    """
    Ranked IRV-style primary until ``n`` candidates remain, then IRV general. [1]_

    Parameters
    ----------
    election : array_like
        Ranked ballots.
    n : int
        Target number of finalists after the primary.
    tiebreaker : {'random', 'order', None}, optional

    Returns
    -------
    winner : {int, None}

    References
    ----------
    .. [1] :wikipedia:`Top-four primary#Variations`
    """
    finalists = irv_eliminate_to_n(election, n, tiebreaker)
    if finalists is None:
        return None
    sub, new_to_old = _restrict_ballots(election, finalists)
    w = irv(sub, tiebreaker)
    if w is None:
        return None
    return int(new_to_old[w])


def irv_primary_top_n_runoff(election, n, tiebreaker=None):
    """
    IRV-style primary until ``n`` remain, then top-two contingent general. [1]_

    Parameters
    ----------
    election : array_like
        Ranked ballots.
    n : int
        Target number of finalists after the primary.
    tiebreaker : {'random', 'order', None}, optional

    Returns
    -------
    winner : {int, None}

    References
    ----------
    .. [1] :wikipedia:`Top-four primary#Variations`
    """
    finalists = irv_eliminate_to_n(election, n, tiebreaker)
    if finalists is None:
        return None
    sub, new_to_old = _restrict_ballots(election, finalists)
    w = runoff(sub, tiebreaker)
    if w is None:
        return None
    return int(new_to_old[w])


def top_four_irv(election, tiebreaker=None):
    """Same as ``top_n_irv(election, 4, tiebreaker)``."""
    return top_n_irv(election, 4, tiebreaker)


def top_five_irv(election, tiebreaker=None):
    """Same as ``top_n_irv(election, 5, tiebreaker)``."""
    return top_n_irv(election, 5, tiebreaker)


def top_four_runoff(election, tiebreaker=None):
    """Same as ``top_n_runoff(election, 4, tiebreaker)``."""
    return top_n_runoff(election, 4, tiebreaker)


def top_five_runoff(election, tiebreaker=None):
    """Same as ``top_n_runoff(election, 5, tiebreaker)``."""
    return top_n_runoff(election, 5, tiebreaker)


def top_four_condorcet(election, tiebreaker=None):
    """Same as ``top_n_condorcet(election, 4, tiebreaker)``."""
    return top_n_condorcet(election, 4, tiebreaker)


def top_five_condorcet(election, tiebreaker=None):
    """Same as ``top_n_condorcet(election, 5, tiebreaker)``."""
    return top_n_condorcet(election, 5, tiebreaker)
