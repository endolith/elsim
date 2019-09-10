import numpy as np
from numpy.testing import assert_array_equal, assert_array_less
import pytest
from elections import random_utilities, impartial_culture


def test_random_utilities():
    for n_voters in (1, 5, 100):
        for n_cands in (1, 2, 3, 7, 100):
            election = random_utilities(n_voters, n_cands)

            # Make sure rows are voters and columns are
            # candidates
            assert election.shape == (n_voters, n_cands)

            # Make sure each row is uniformly distributed
            for row in election:
                assert row.min() >= 0
                assert row.max() <= 1

    # Check that utilities are equally distributed
    np.random.seed(42)
    n_voters = 10000
    n_cands = 15
    bins = 10
    election = random_utilities(n_voters, n_cands)
    hist = np.histogram(election.flatten(), bins=bins, range=(0, 1))[0]
    assert_array_less(hist/(n_voters*n_cands), 0.11)


def test_impartial_culture():
    for n_voters in (0, 1, 5, 100):
        for n_cands in (0, 1, 2, 3, 7, 100):
            election = impartial_culture(n_voters, n_cands)

            # Make sure rows are voters and columns are
            # rankings (= same number as candidates)
            assert election.shape == (n_voters, n_cands)

            # Make sure each row is a permutation with no ties
            for row in election:
                # (Empty arrays test equal, which is fine)
                assert_array_equal(np.bincount(row), 1)

    # Check that rankings are equally distributed
    np.random.seed(42)
    n_voters = 10000
    n_cands = 15
    election = impartial_culture(n_voters, n_cands)
    for col in election.T:
        assert_array_less(abs(np.bincount(col)/(n_voters/n_cands) - 1),
                          0.15)


def test_invalid():
    with pytest.raises(ValueError):
        impartial_culture(1, 257)


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
