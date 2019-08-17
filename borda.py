import numpy as np
import pytest


def borda(election, tiebreaker=None):
    """
    Finds the winner of a ranked ballot election using the Borda count method

    Parameters
    ----------
    election : array_like
        A collection of ranked ballots.
        Rows represent voters and columns represent rankings, from best to
        worst, with no tied rankings.
        Each cell contains the ID number of a candidate, starting at 0.

        For example, if a voter ranks Curie > Avogadro > Bohr, the ballot line
        would read ``[2, 0, 1]`` (with IDs in alphabetical order).
    tiebreaker : {'random', None}, optional
        If there is a tie, and `tiebreaker` is ``'random'``, a random finalist
        is returned.  By default, ``None`` is returned for ties.

    Returns
    -------
    winner : int
        The ID number of the winner, or ``None`` for an unbroken tie.
    """
    election = np.asarray(election)

    ncands = election.shape[1]
    total_tally = np.zeros(ncands, dtype=int)

    # Tally candidates in each column, multiply by points for each rank level
    for n, column in enumerate(election.T):
        tally = np.bincount(column, minlength=ncands)
        total_tally += (ncands - n)*tally

    # Python lists are faster than NumPy here
    total_tally = list(total_tally)
    highest = max(total_tally)
    if tiebreaker == 'random':
        return total_tally.index(highest)
    elif tiebreaker is None:
        n_winners = total_tally.count(highest)
        if n_winners == 1:
            return total_tally.index(highest)
        elif n_winners > 1:
            # There is a tie
            return None
        else:
            raise RuntimeError('Bug in Borda count')
    else:
        raise ValueError('Tiebreaker not understood')


def test_borda_basic():
    # Manually calculated correct answer
    election = [[0, 1, 4, 3, 2],
                [4, 2, 3, 1, 0],
                [4, 2, 3, 1, 0],
                [3, 2, 1, 4, 0],
                [2, 0, 3, 1, 4],
                [3, 2, 1, 4, 0],
                ]

    assert borda(election) == 2

    # Example from
    # https://www3.nd.edu/~apilking/math10170/information/Lectures/Lecture-2.Borda%20Method.pdf
    K, H, R = 0, 1, 2
    election = [*2*[[K, H, R]],
                *3*[[H, R, K]],
                *2*[[H, K, R]],
                *3*[[R, H, K]],
                ]

    assert borda(election) == H

    # Example from
    # http://jlmartin.faculty.ku.edu/~jlmartin/courses/math105-F11/Lectures/chapter1-part2.pdf
    A, B, C, D = 0, 1, 2, 3
    election = [*14*[[A, B, C, D]],
                *10*[[C, B, D, A]],
                * 8*[[D, C, B, A]],
                * 4*[[B, D, C, A]],
                * 1*[[C, D, B, A]],
                ]

    assert borda(election) == B

    election = [*60*[[A, B, C, D]],
                *40*[[B, D, C, A]],
                ]

    assert borda(election) == B

    # Example from
    # http://www.yorku.ca/bucovets/4380/exercises/exercises_1_a.pdf
    v, w, x, y, z = 0, 1, 2, 3, 4
    election = [*11*[[v, w, x, y, z]],
                *12*[[w, x, y, z, v]],
                *13*[[x, v, w, y, z]],
                *14*[[y, w, v, z, x]],
                *15*[[z, v, x, w, y]],
                ]

    assert borda(election) == w


def test_borda_invalid():
    with pytest.raises(ValueError):
        election = [[0, 1, 4, 3, 2],
                    [4, 2, 3, 1, 0],
                    ]

        borda(election, 'dictator')


if __name__ == "__main__":
    # Run unit tests, in separate process to avoid warnings about cached
    # modules, printing output line by line in realtime
    from subprocess import Popen, PIPE
    with Popen(['pytest',
                '--tb=short',  # shorter traceback format
                '--hypothesis-show-statistics',
                str(__file__)], stdout=PIPE, bufsize=1,
               universal_newlines=True) as p:
        for line in p.stdout:
            print(line, end='')
