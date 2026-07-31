import numpy as np

from elsim.methods._common import (_all_indices, _get_tiebreak, _no_tiebreak,
                                   _order_tiebreak_keep, _random_tiebreak)

_tiebreak_map = {'order': _order_tiebreak_keep,
                 'random': _random_tiebreak,
                 None: _no_tiebreak}


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
    election = np.asarray(election)

    ncands = election.shape[1]
    total_tally = np.zeros(ncands, dtype=int)

    # Tally candidates in each column, multiply by points for each rank level
    for n, column in enumerate(election.T):
        tally = np.bincount(column, minlength=ncands)
        total_tally += (ncands - n)*tally

    # Python lists are faster than NumPy here
    total_tally = total_tally.tolist()

    # Find the set of candidates who have the highest score (usually only one)
    highest = max(total_tally)
    winners = _all_indices(total_tally, highest)

    # Break any ties using specified method
    tiebreak = _get_tiebreak(tiebreaker, _tiebreak_map)
    return tiebreak(winners)[0]
