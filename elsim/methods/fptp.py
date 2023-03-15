import numpy as np
from elsim.methods._common import (_all_indices, _order_tiebreak_keep,
                                   _random_tiebreak, _no_tiebreak)

_tiebreak_map = {'order': _order_tiebreak_keep,
                 'random': _random_tiebreak,
                 None: _no_tiebreak}


def _get_tiebreak(tiebreaker):
    try:
        return _tiebreak_map[tiebreaker]
    except KeyError:
        raise ValueError('Tiebreaker not understood')


def fptp(election, tiebreaker=None):
    """
    Find the winner of an election using first-past-the-post / plurality rule.

    The candidate with the largest number of first preferences wins.[1]_

    Parameters
    ----------
    election : array_like
        A 2D collection of ranked ballots.  (See `borda` for election format.)
        Or a 1D array of first preferences only.
    tiebreaker : {'random', 'order', None}, optional
        If there is a tie, and `tiebreaker` is ``'random'``, a random finalist
        is returned.
        If 'order', the lowest-ID tied candidate is returned.
        By default, ``None`` is returned for ties.

    Returns
    -------
    winner : int
        The ID number of the winner, or ``None`` for an unbroken tie.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Plurality_voting

    Examples
    --------
    Label some candidates:

    >>> A, B, C = 0, 1, 2

    Specify the ballots for the 5 voters:

    >>> election = [[A, C, B],
                    [A, C, B],
                    [B, A, C],
                    [B, C, A],
                    [B, C, A],
                    [C, A, B],
                    ]

    Candidate B (1) gets the most first-preference votes, and is the winner:

    >>> fptp(election)
    1

    Single-mark ballots can also be tallied (with ties broken as specified):

    >>> election = [A, B, B, C, C]
    >>> print(fptp(election))
    None

    There is a tie between B (1) and C (2).  ``tiebreaker=order`` always
    prefers the lower-numbered candidate in a tie:

    >>> fptp(election, 'order')
    1
    """
    election = np.asarray(election)

    # Get first preferences from election array
    if election.ndim == 2:
        first_preferences = election[:, 0]
    elif election.ndim == 1:
        first_preferences = election
    else:
        raise ValueError('Election array must be 2D ranked ballots or 1D'
                         'list of first preferences')

    # Tally all first preferences (with index of tally = candidate ID)
    tallies = np.bincount(first_preferences)

    # Find the set of candidates who have the highest tally
    highest_tally = max(tallies)
    winners = _all_indices(tallies, highest_tally)

    # Break any ties using specified method
    tiebreak = _get_tiebreak(tiebreaker)
    return tiebreak(winners)[0]


def sntv(election, n=1, tiebreaker=None):
    """
    Find winners of an election using the single non-transferable vote rule.

    This is a multi-winner generalization of `fptp`.
    The candidates with the largest number of first preferences win.[1]_

    Parameters
    ----------
    election : array_like
        A 2D collection of ranked ballots.  (See `borda` for election format.)
        Or a 1D array of first preferences only.
    tiebreaker : {'random', 'order', None}, optional
        If there is a tie, and `tiebreaker` is ``'random'``, random tied
        candidates are returned.
        If 'order', the lowest-ID tied candidates are returned.
        By default, ``None`` is returned for ties.

    Returns
    -------
    winner : set
        The ID numbers of the winners, or ``None`` for an unbroken tie.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Single_non-transferable_vote

    Examples
    --------
    Label some candidates:

    >>> A, B, C, D = 0, 1, 2, 3

    Specify the ballots for the 5 voters:

    >>> election = [[A, C, B],
                    [A, C, B],
                    [B, A, C],
                    [B, C, A],
                    [B, C, A],
                    [C, A, B],
                    ]

    Candidate B (1) gets the most first-preference votes, and Candidate A (1)
    comes in second.  If SNTV is electing two candidates, A and B will win:

    >>> sntv(election, 2)
    {0, 1}

    Single-mark ballots can also be tallied (with ties broken as specified):

    >>> election = [A, A, B, B, C, C, D]
    >>> print(sntv(election, 2))
    None

    There is a tie between A (0), B (1) and C (2), and no tiebreaker is
    specified.  If instead we use ``tiebreaker=order``, it always prefers the
    lower-numbered candidates in a tie:

    >>> sntv(election, 2, 'order')
    {0, 1}
    """
    election = np.asarray(election)

    # Get first preferences from election array
    if election.ndim == 2:
        first_preferences = election[:, 0]
    elif election.ndim == 1:
        first_preferences = election
    else:
        raise ValueError('Election array must be 2D ranked ballots or 1D'
                         'list of first preferences')

    # Tally all first preferences (with index of tally = candidate ID)
    tallies = np.bincount(first_preferences)

    if len(np.nonzero(tallies)[0]) <= n:
        return set(first_preferences)

    # Find the set of candidates who have the highest tally
    sorted_indices = np.argsort(tallies)
    top_candidates = sorted_indices[-n:]

    # Does the worst candidate in this set have the same votes as any
    # candidate outside this set?  Ties only need to be broken between
    # candidates who have that many votes.  top_candidates is sorted, so:
    worst_top = top_candidates[0]
    best_not_top = sorted_indices[-n-1]

    # There might be more than one with the same tally.  Or some with one tally
    # and some with another
    if tallies[worst_top] == tallies[best_not_top]:
        # There is a tie that extends beyond top candidates

        # Finalists who are not tied with a non-finalist
        untied_winners = sorted_indices[tallies[sorted_indices] >
                                        tallies[worst_top]]

        # Finalists and non-finalists who are tied
        tied_candidates = np.where(tallies == tallies[worst_top])[0]

        # Break any ties using specified method
        tiebreak = _get_tiebreak(tiebreaker)
        n_needed = n - len(untied_winners)
        tie_winners = tiebreak(list(tied_candidates), n_needed)
        if tie_winners == [None]:
            return None
        return set(untied_winners) | set(tie_winners)
    else:
        # TODO: Maybe should use arrays for deterministic randomness?
        return set(top_candidates)
