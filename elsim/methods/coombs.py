import numpy as np

from elsim.methods._common import (_all_indices, _dec_rank_idx, _get_tiebreak,
                                   _inc_rank_idx, _no_tiebreak,
                                   _order_tiebreak_elim, _random_tiebreak,
                                   _tally_at_rank_idx)

_tiebreak_map = {'order': _order_tiebreak_elim,
                 'random': _random_tiebreak,
                 None: _no_tiebreak}


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
    .. [1] https://en.wikipedia.org/wiki/Coombs%27_method

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
    election = np.asarray(election)
    n_voters, n_cands = election.shape
    tiebreak = _get_tiebreak(tiebreaker, _tiebreak_map)
    voter_top_rank_idx = np.zeros(n_voters, dtype=np.uint8)
    cand_top_tallies = np.empty(n_cands, dtype=np.uint)
    voter_bottom_rank_idx = np.full(n_voters, n_cands - 1, dtype=np.uint8)
    cand_bottom_tallies = np.empty(n_cands, dtype=np.uint)

    # No IRV-style eager elimination of zero top tallies here: Coombs'
    # elimination criterion is last-place tallies, not lowest top tallies.

    eliminated_cands = set()

    for round_ in range(n_cands):
        _tally_at_rank_idx(cand_top_tallies, election, voter_top_rank_idx)

        # (tolist makes things 2-4x faster)
        cand_top_tallies_list = cand_top_tallies.tolist()

        # Did anyone get a majority?
        max_cand_top_tally = max(cand_top_tallies_list)
        if max_cand_top_tally > n_voters / 2:
            return cand_top_tallies_list.index(max_cand_top_tally)

        # If not, eliminate candidate with the most last-place votes
        _tally_at_rank_idx(cand_bottom_tallies, election,
                           voter_bottom_rank_idx)
        # (tolist makes things 2-4x faster)
        cand_bottom_tallies_list = cand_bottom_tallies.tolist()
        max_cand_bottom_tally = max(cand_bottom_tallies_list)
        max_bottom_tally_cands = _all_indices(cand_bottom_tallies_list,
                                              max_cand_bottom_tally)
        cand_to_eliminate = tiebreak(max_bottom_tally_cands)[0]

        # Handle no tiebreaker case
        if cand_to_eliminate is None:
            return None

        # Eliminate candidate with highest last-preference tally
        eliminated_cands.add(cand_to_eliminate)

        # Increment/decrement rank indices past all eliminated candidates
        _inc_rank_idx(election, voter_top_rank_idx, eliminated_cands)
        _dec_rank_idx(election, voter_bottom_rank_idx, eliminated_cands)

        # (top and bottom rank indices move in opposite directions)
    raise RuntimeError("Bug in Coombs' calculation")
