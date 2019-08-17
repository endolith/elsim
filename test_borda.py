import pytest
from borda import borda


def test_basic():
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


def test_invalid():
    with pytest.raises(ValueError):
        election = [[0, 1]]
        borda(election, tiebreaker='dictator')


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
