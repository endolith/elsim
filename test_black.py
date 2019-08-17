import numpy as np
from black import black


def test_condorcet_winner():
    # Example from
    # https://en.wikipedia.org/wiki/Condorcet_method#Pairwise_counting_and_matrices
    election = np.array([[1, 2, 0, 3],
                         [3, 0, 2, 1],
                         [0, 2, 1, 3],
                         ])
    assert black(election) == 0

    # Example from https://electowiki.org/wiki/Condorcet_Criterion
    election = np.concatenate((np.tile([0, 1, 2], (499, 1)),
                               np.tile([2, 1, 0], (498, 1)),
                               np.tile([1, 2, 0], (3, 1)),
                               ))
    assert black(election) == 1

    # Example from
    # https://www3.nd.edu/~apilking/Math10170/Information/Lectures/Lecture_3.Head%20To%20Head%20Comparisons.pdf
    Colley, Henry, Taylor = 0, 1, 2
    election = np.array([[Colley,  Henry, Taylor],
                         [ Henry, Colley, Taylor],
                         [ Henry, Colley, Taylor],
                         [Taylor, Colley,  Henry],
                         [Taylor,  Henry, Colley],
                         ])
    assert black(election) == 1

    # Example from https://www.whydomath.org/node/voting/impossible.html
    election = np.array([[0, 2, 3, 1],
                         [1, 2, 3, 0],
                         [3, 0, 2, 1],
                         [0, 1, 3, 2],
                         [3, 0, 2, 1],
                         ])
    assert black(election) == 3

    # Table 3.1 from Mackie - Democracy Defended
    # (Borda and Condorcet results differ)
    A, B, C, D, E = 0, 1, 2, 3, 4
    election = [*4*[[A, E, D, C, B]],
                *3*[[B, C, E, D, A]],
                *2*[[C, D, E, B, A]],
                ]

    assert black(election) == C  # "and C is the Condorcet winner"


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
