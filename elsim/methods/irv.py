import numpy as np
from elsim.methods._common import (_all_indices, _tally_at_pointer,
                                   _inc_pointer, _order_tiebreak_elim,
                                   _random_tiebreak, _no_tiebreak)

_tiebreak_map = {'order': _order_tiebreak_elim,
                 'random': _random_tiebreak,
                 None: _no_tiebreak}


def _get_tiebreak(tiebreaker):
    try:
        return _tiebreak_map[tiebreaker]
    except KeyError:
        raise ValueError('Tiebreaker not understood')


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
    tiebreaker : {'random', None}, optional
        If there is a tie, and `tiebreaker` is ``'random'``, tied candidates
        are eliminated or selected at random
        is returned.
        If 'order', the lowest-ID tied candidate is preferred in each tie.
        By default, ``None`` is returned if there are any ties.

    Returns
    -------
    winner : int
        The ID number of the winner, or ``None`` for an unbroken tie.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Instant-runoff_voting
    .. [2] https://en.wikipedia.org/wiki/Exhaustive_ballot

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
    election = np.asarray(election)
    n_voters = election.shape[0]
    n_cands = election.shape[1]
    eliminated = set()
    tiebreak = _get_tiebreak(tiebreaker)
    first_pointer = np.zeros(n_voters, dtype=np.uint8)
    first_tallies = np.empty(n_cands, dtype=np.uint)
    for _ in range(n_cands):
        _tally_at_pointer(first_tallies, election, first_pointer)

        # tolist makes things 2-4x faster
        first_tallies_list = first_tallies.tolist()

        # Did anyone get a majority?
        highest = max(first_tallies_list)
        if highest > n_voters / 2:
            return first_tallies_list.index(highest)

        # If not, eliminate lowest
        lowest = min(x for x in first_tallies_list if x != 0)  # faster?
        low_scorers = _all_indices(first_tallies_list, lowest)
        loser = tiebreak(low_scorers)[0]

        # Handle no tiebreaker case
        if loser is None:
            return None

        # Add candidate with lowest score in this round
        eliminated.add(loser)

        # Make sure candidates who never got votes are also eliminated
        # TODO: In the future when round tallies are also output, this should
        # be its own round
        eliminated.update(_all_indices(first_tallies_list, 0))

        # Increment pointers until they point at non-eliminated candidates
        _inc_pointer(election, first_pointer, eliminated)

    raise RuntimeError('Bug in IRV calculation')
