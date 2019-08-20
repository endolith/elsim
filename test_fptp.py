from fptp import fptp
import numpy as np
import pytest


def test_basic():
    # Standard Tennessee example
    # https://en.wikipedia.org/wiki/Template:Tenn_voting_example
    Memphis, Nashville, Chattanooga, Knoxville = 0, 1, 2, 3
    election = [*42*[[Memphis, Nashville, Chattanooga, Knoxville]],
                *26*[[Nashville, Chattanooga, Knoxville, Memphis]],
                *15*[[Chattanooga, Knoxville, Nashville, Memphis]],
                *17*[[Knoxville, Chattanooga, Nashville, Memphis]],
                ]

    assert fptp(election) == Memphis

    # Example from Ques 9
    # http://www.yorku.ca/bucovets/4380/exercises/exercises_1_a.pdf
    v, w, x, y, z = 0, 1, 2, 3, 4
    election = [*11*[[v, w, x, y, z]],
                *12*[[w, x, y, z, v]],
                *13*[[x, v, w, y, z]],
                *14*[[y, w, v, z, x]],
                *15*[[z, v, x, w, y]],
                ]

    assert fptp(election) == z

    election = np.array([[2, 3, 1, 0],
                         [0, 1, 2, 3],
                         [2, 1, 3, 0],
                         [1, 0, 3, 2]])
    assert fptp(election) == 2

    election = np.array([[2, 0, 1],
                         [0, 1, 2],
                         [2, 0, 1],
                         [1, 2, 0],
                         [0, 1, 2],
                         [2, 0, 1],
                         [1, 0, 2],
                         [2, 0, 1],
                         [1, 0, 2],
                         [0, 2, 1]])
    assert fptp(election) == 2

    # Example from
    # http://jlmartin.faculty.ku.edu/~jlmartin/courses/math105-F11/Lectures/chapter1-part2.pdf
    A, B, C, D = 0, 1, 2, 3
    election = [*60*[[A, B, C, D]],
                *40*[[B, D, C, A]],
                ]

    assert fptp(election) == A


def test_invalid():
    with pytest.raises(ValueError):
        election = [[0, 1]]
        fptp(election, 'dictator')
