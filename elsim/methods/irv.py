import numpy as np

from elsim.methods._common import (_all_indices, _get_tiebreak, _no_tiebreak,
                                   _order_tiebreak_elim, _random_tiebreak,
                                   _tally_at_rank_idx)
from elsim.methods._runoff import (remaining_minimum, run_elimination_rounds,
                                   winner_from_result)

_tiebreak_map = {'order': _order_tiebreak_elim,
                 'random': _random_tiebreak,
                 None: _no_tiebreak}


def _zero_vote_candidates(election):
    """Mask of candidates who receive no first-preference votes at all."""
    n_voters, n_cands = election.shape
    tallies = np.empty(n_cands, dtype=np.uint)
    _tally_at_rank_idx(tallies, election, np.zeros(n_voters, dtype=np.uint8))
    mask = np.zeros(n_cands, dtype=bool)
    mask[_all_indices(tallies.tolist(), 0)] = True
    return mask


def _irv_losers(election, eliminated_mask, top_tallies):
    """IRV rule: the remaining candidate(s) with the fewest first choices."""
    tied = remaining_minimum(top_tallies.tolist(), eliminated_mask)
    return tied, {'tallies_before': top_tallies.copy()}


def irv_rounds(election, tiebreaker=None, *, min_remaining=1,
               record_rounds=False):
    """
    Run instant-runoff voting and return per-round elimination data.

    Parameters
    ----------
    election : array_like
        Ranked ballots; see `irv`.
    tiebreaker : {'random', 'order', None}, optional
        Tie-breaking rule; see `irv`.
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
    tiebreak = _get_tiebreak(tiebreaker, _tiebreak_map)

    # Candidates with no first-choice votes are dropped as a batch before the
    # rounds begin, so several of them never produce an elimination "tie".
    # TODO: When round tallies are also output, this should be its own round
    # (or a batch_elimination=True option).
    return run_elimination_rounds(
        election, _irv_losers, tiebreak,
        min_remaining=min_remaining,
        record_rounds=record_rounds,
        eliminated_mask=_zero_vote_candidates(election),
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
    return winner_from_result(irv_rounds(election, tiebreaker=tiebreaker),
                              'IRV')
