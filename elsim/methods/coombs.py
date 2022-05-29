import numpy as np
from ._common import (_all_indices, _tally_at_pointer, _inc_pointer,
                      _dec_pointer, _order_tiebreak_elim, _random_tiebreak,
                      _no_tiebreak)

_tiebreak_map = {'order': _order_tiebreak_elim,
                 'random': _random_tiebreak,
                 None: _no_tiebreak}


def _get_tiebreak(tiebreaker):
    try:
        return _tiebreak_map[tiebreaker]
    except KeyError:
        raise ValueError('Tiebreaker not understood')


def coombs(election, tiebreaker=None):
    """
    Find the winner of an election using Coomb's method.

    If any candidate gets a majority of first-preference votes, they win.
    Otherwise, the candidate(s) with the most number of last-preference votes
    is eliminated, votes for eliminated candidates are transferred according
    to the voters' preference rankings, and a series of runoff elections are
    held between the remainders until a candidate gets a majority.[1]_

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
    .. [1] https://en.wikipedia.org/wiki/Coombs%27_method

    Examples
    --------
    Label some candidates:

    >>> A, B, C = 0, 1, 2

    Specify the ballots for the 5 voters:

    >>> election = [[A, C, B],
                    [A, C, B],
                    [B, C, A],
                    [B, C, A],
                    [C, A, B],
                    ]

    In the first round, no candidate gets a majority, so Candidate B (1) is
    eliminated, for receiving 3 out of 5 last-place votes.  Voter 2 and 3's
    support of B is transferred to Candidate C (2), causing Candidate C to win,
    with 3 out of 5 votes:

    >>> coombs(election)
    2
    """
    election = np.asarray(election)
    n_voters = election.shape[0]
    n_cands = election.shape[1]
    eliminated = set()
    tiebreak = _get_tiebreak(tiebreaker)
    first_pointer = np.zeros(n_voters, dtype=np.uint8)
    first_tallies = np.empty(n_cands, dtype=np.uint)
    last_pointer = np.full(n_voters, n_cands - 1, dtype=np.uint8)
    last_tallies = np.empty(n_cands, dtype=np.uint)
    for round_ in range(n_cands):
        _tally_at_pointer(first_tallies, election, first_pointer)

        # tolist makes things 2-4x faster
        first_tallies_list = first_tallies.tolist()

        # Did anyone get majority
        highest = max(first_tallies_list)
        if highest > n_voters / 2:
            return first_tallies_list.index(highest)

        # If not, eliminate candidate with highest number of last-preferences
        _tally_at_pointer(last_tallies, election, last_pointer)
        highest = max(last_tallies)
        highly_hated = _all_indices(last_tallies, highest)

        loser = tiebreak(highly_hated)[0]

        # Handle no tiebreaker case
        if loser is None:
            return None

        # Add candidate with lowest score in this round
        eliminated.add(loser)

        # Increment pointers past all eliminated candidates
        _inc_pointer(election, first_pointer, eliminated)
        _dec_pointer(election, last_pointer, eliminated)

        # low and high pointer need to increment opposite
    raise RuntimeError("Bug in Coombs' calculation")
