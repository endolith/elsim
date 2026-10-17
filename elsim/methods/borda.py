import numpy as np

from elsim.methods._common import (_all_indices, _get_tiebreak, _no_tiebreak,
                                   _order_tiebreak_keep, _random_tiebreak)

_tiebreak_map = {'order': _order_tiebreak_keep,
                 'random': _random_tiebreak,
                 None: _no_tiebreak}


def borda_tallies(election, eliminated_mask=None, *, first_place_points=None):
    """
    Positional (Borda) point totals, optionally ignoring eliminated candidates.

    Each ballot awards ``first_place_points`` to its highest-ranked remaining
    candidate, one point fewer to the next, and so on.  Eliminated candidates
    are skipped as if they had never been on the ballot and always total 0.

    Parameters
    ----------
    election : array_like
        Ranked ballots (n_voters × n_cands).  See `borda` for the format.
    eliminated_mask : array_like of bool, optional
        ``eliminated_mask[i]`` is True if candidate ``i`` is out of the race.
        By default nobody is eliminated.
    first_place_points : int, optional
        Points for a ballot's top remaining candidate.  Defaults to the number
        of remaining candidates (1-based convention: last place scores 1).
        Pass ``n_remaining - 1`` for the 0-based convention used by
        Baldwin's method / TVR.

    Returns
    -------
    tallies : ndarray of float, length n_cands
        Total points per candidate.
    """
    election = np.asarray(election)
    n_cands = election.shape[1]

    if eliminated_mask is None:
        active = np.ones(election.shape, dtype=bool)
        n_remaining = n_cands
    else:
        eliminated_mask = np.asarray(eliminated_mask, dtype=bool)
        active = ~eliminated_mask[election]
        n_remaining = n_cands - int(np.count_nonzero(eliminated_mask))

    if first_place_points is None:
        first_place_points = n_remaining

    # Position of every cell among the *remaining* candidates on its ballot.
    positions = np.cumsum(active, axis=1) - 1
    points = (first_place_points - positions)[active]
    return np.bincount(election[active], weights=points, minlength=n_cands)


def borda(election, tiebreaker=None):
    """
    Find the winner of a ranked ballot election using the Borda count method.

    Borda's original formulation gives the lowest-ranked candidate 1 point,
    second-lowest 2 points, and so on.  Borda noted that starting at 0 yields
    the same result for complete ballots, and modern papers and methods (TVR,
    Emerson Modified Borda Count) often use the 0-based convention: ``n - 1``
    points for 1st place, 0 for last. [1]_

    This implementation uses 1-based scoring (``n`` points for 1st, 1 for
    last), which is equivalent to 0-based for complete ballots up to a
    constant shift of ``n_voters`` per candidate.  See `baldwin_rounds` for an
    example of a method that uses the 0-based convention internally.

    Parameters
    ----------
    election : array_like
        A collection of ranked ballots.
        Rows represent voters and columns represent rankings, from best to
        worst, with no tied rankings.
        Each cell contains the ID number of a candidate, starting at 0.

        For example, if a voter ranks Curie > Avogadro > Bohr, the ballot line
        would read ``[2, 0, 1]`` (with IDs in alphabetical order).
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
    .. [1] :wikipedia:`Borda count`

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

    Candidate A gets a total of 3+3+1+1+2 = 10 points.
    Candidate B gets a total of 1+1+3+3+1 =  9 points.
    Candidate C gets a total of 2+2+2+2+3 = 11 points.
    Candidate C is the winner:

    >>> borda(election)
    2

    """
    # Point totals are exact integers; round away any float noise before
    # comparing.  (Python lists are faster than NumPy for the small tail.)
    total_tally = np.rint(borda_tallies(election)).astype(int).tolist()

    highest = max(total_tally)
    winners = _all_indices(total_tally, highest)

    tiebreak = _get_tiebreak(tiebreaker, _tiebreak_map)
    return tiebreak(winners)[0]
