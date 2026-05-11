import numpy as np

from elsim.methods._common import (
    _all_indices,
    _get_tiebreak,
    _inc_rank_idx,
    _no_tiebreak,
    _order_tiebreak_keep,
    _random_tiebreak,
    _tally_at_rank_idx,
)

_tiebreak_map = {'order': _order_tiebreak_keep,
                 'random': _random_tiebreak,
                 None: _no_tiebreak}


def runoff(election, tiebreaker=None):
    """
    Find the winner of an election using top-two runoff / two-round system.

    If any candidate gets a majority of first-preference votes, they win.
    Otherwise, a runoff election is held between the two highest-voted
    candidates. [1]_

    The votes in the first and second rounds are calculated from the same set
    of ranked ballots, so this is actually the contingent vote method. [2]_  If
    voters are honest and consistent between rounds, then the two methods are
    equivalent.  It can also be considered "top-two IRV", and is closely
    related to the supplementary vote (which restricts rankings to two).

    Parameters
    ----------
    election : array_like
        A collection of ranked ballots.  See `borda` for election format.
        Currently, this must include full rankings for each voter.
    tiebreaker : {'random', 'order', None}, optional
        If there is a tie, and `tiebreaker` is ``'random'``, a random finalist
        is returned.
        If 'order', the lowest-ID tied candidate is returned.
        By default, ``None`` is returned for ties.

    Returns
    -------
    winner : {int, None}
        The ID number of the winner, or ``None`` for an unbroken tie.

    References
    ----------
    .. [1] :wikipedia:`Two-round system`
    .. [2] :wikipedia:`Contingent vote`

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
    eliminated.  A runoff is held between the top two: Candidates A (0) and
    B (1).   Voter 4's support is transferred to Candidate A, causing
    Candidate A to win, with 3 out of 5 votes:

    >>> runoff(election)
    0
    """
    election = np.asarray(election)
    n_voters = election.shape[0]
    first_preferences = election[:, 0]
    tallies = np.bincount(first_preferences).tolist()
    highest = sorted(tallies)[-2:]
    high_scorers = _all_indices(tallies, highest[-1])

    tiebreak = _get_tiebreak(tiebreaker, _tiebreak_map)
    finalists = tiebreak(high_scorers, 2)
    # Handle no tiebreaker case
    if None in finalists:
        return None

    if len(finalists) == 1:
        if tallies[finalists[0]] > n_voters / 2:
            return finalists[0]
        second_scorers = _all_indices(tallies, highest[0])
        finalists += tiebreak(second_scorers)

    # So at this point we should have two finalists who were chosen according
    # to tiebreaker rules.
    # Possibly including Nones

    # Handle no tiebreaker case
    if None in finalists:
        return None

    finalist_0 = finalists[0]
    finalist_1 = finalists[1]

    n_cands = election.shape[1]
    cand_tallies = np.empty(n_cands, dtype=np.uint)
    voter_top_rank_idx = np.zeros(n_voters, dtype=np.uint8)
    eliminated_cands = set(range(n_cands)) - {finalist_0, finalist_1}
    if eliminated_cands:
        _inc_rank_idx(election, voter_top_rank_idx, eliminated_cands)
    _tally_at_rank_idx(cand_tallies, election, voter_top_rank_idx)
    finalist_0_tally = cand_tallies[finalist_0]
    finalist_1_tally = cand_tallies[finalist_1]

    assert finalist_0_tally + finalist_1_tally == n_voters

    if finalist_0_tally == finalist_1_tally:
        return tiebreak([finalist_0, finalist_1])[0]
    elif finalist_0_tally > finalist_1_tally:
        return finalist_0
    else:
        return finalist_1
