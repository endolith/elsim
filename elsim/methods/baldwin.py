import numpy as np

from elsim.methods._common import (_get_tiebreak, _no_tiebreak,
                                   _order_tiebreak_elim, _random_tiebreak)
from elsim.methods._runoff import (remaining_minimum, run_elimination_rounds,
                                   winner_from_result)
from elsim.methods.borda import borda_tallies

_tiebreak_map = {'order': _order_tiebreak_elim,
                 'random': _random_tiebreak,
                 None: _no_tiebreak}


def _tvr_scores(election, eliminated_mask):
    """
    Total votes among remaining candidates, TVR convention
    (``n_remaining - 1`` points for 1st place, 0 for last).
    """
    n_remaining = election.shape[1] - int(np.count_nonzero(eliminated_mask))
    return borda_tallies(election, eliminated_mask,
                         first_place_points=n_remaining - 1)


def _promoted_per_voter(election, eliminated_mask, loser):
    """
    For each voter, the remaining candidates ranked below ``loser``.

    When the loser is removed, each of these moves up one rank on that
    ballot.  In the TVR convention (points = number of remaining candidates
    ranked below you) their score is unchanged while every *other* survivor
    loses one point per voter; equivalently, in the pre-elimination scale
    the promoted candidates gain one point each.  ``eliminated_mask`` must
    already include the loser.

    Returns
    -------
    promoted : list of ndarray, length n_voters
    """
    n_cands = election.shape[1]
    loser_position = np.argmax(election == loser, axis=1)
    below_loser = np.arange(n_cands) > loser_position[:, None]
    promoted_mask = below_loser & ~eliminated_mask[election]
    return [ballot[mask] for ballot, mask in zip(election, promoted_mask)]


def _baldwin_losers(election, eliminated_mask, top_tallies):
    """Baldwin rule: the remaining candidate(s) with the lowest score."""
    scores = _tvr_scores(election, eliminated_mask)
    tied = remaining_minimum(scores.tolist(), eliminated_mask)
    return tied, {'borda_before': scores}


def _baldwin_round_extras(election, eliminated_mask, loser):
    """Extra per-round animation data recorded after the loser is removed."""
    return {
        'borda_after': _tvr_scores(election, eliminated_mask),
        'promoted_per_voter': _promoted_per_voter(election, eliminated_mask,
                                                  loser),
    }


def baldwin_rounds(election, tiebreaker=None, *, min_remaining=1,
                   record_rounds=False):
    """
    Run Baldwin's method (Total Vote Runoff) and return per-round
    elimination data.

    Baldwin's method re-tallies total votes each round among remaining
    candidates only and eliminates the lowest scorer.  A candidate with a
    majority of first-preference votes wins immediately.

    Total votes follow the TVR convention (also used in the original
    Baldwin method): ``n - 1`` points for 1st place, 0 for last place
    among remaining candidates.  This is the same ranking as the 1-based
    `borda` scoring (``n`` points for 1st, 1 for last) for complete
    ballots -- only a constant shift of ``n_voters`` per candidate -- so
    elimination orders and winners are identical. [1]_ [2]_

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
        total-vote snapshots and promoted-voter data for animation.

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
            ``affected_voters``, ``borda_before``, ``borda_after`` and
            ``promoted_per_voter``.  ``borda_before``/``borda_after`` hold
            the total-vote tallies (TVR convention) before and after the
            elimination; ``promoted_per_voter[v]`` lists the remaining
            candidates that moved up one rank on voter ``v``'s ballot.
            Without ``record_rounds``, each dict has only ``loser``.
        eliminated_mask : ndarray of bool
            Final elimination state.
        final_ballots : ndarray
            Per-voter top remaining choice at end (length n_voters).
        final_tallies : ndarray
            First-preference tallies at end (length n_cands).

    References
    ----------
    .. [1] :wikipedia:`Nanson's method#Baldwin method`
    .. [2] Richard B. L. Foley, "Total Vote Runoff: A
           Majority-Maximizing Form of Ranked Choice Voting",
           *SSRN* 4328946, 2023.
           https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4328946
    """
    tiebreak = _get_tiebreak(tiebreaker, _tiebreak_map)
    return run_elimination_rounds(
        np.asarray(election), _baldwin_losers, tiebreak,
        min_remaining=min_remaining,
        record_rounds=record_rounds,
        round_extras=_baldwin_round_extras,
    )


def baldwin(election, tiebreaker=None):
    """
    Find the winner of an election using Baldwin's method (Total Vote Runoff).

    Total votes are re-tallied each round among remaining candidates only
    (TVR convention: ``n - 1`` points for 1st place, 0 for last), and the
    candidate with the lowest score is eliminated.  If any candidate has a
    majority of first-preference votes at any point, they win
    immediately. [1]_

    See `baldwin_rounds` for details on the scoring convention.

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
    .. [2] Richard B. L. Foley, "Total Vote Runoff: A
           Majority-Maximizing Form of Ranked Choice Voting",
           *SSRN* 4328946, 2023.
           https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4328946

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
    return winner_from_result(
        baldwin_rounds(election, tiebreaker=tiebreaker), "Baldwin's")
