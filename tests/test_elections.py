import numpy as np
from numpy.testing import (assert_array_equal, assert_array_less,
                           assert_allclose)
import pytest
from elsim.elections import (random_utilities, impartial_culture,
                             normal_electorate, normed_dist_utilities)


def test_random_utilities():
    rng = np.random.default_rng(1788)  # Deterministic tests

    for n_voters in (1, 5, 100):
        for n_cands in (1, 2, 3, 7, 100):
            election = random_utilities(n_voters, n_cands, random_state=rng)

            # Make sure rows are voters and columns are
            # candidates
            assert election.shape == (n_voters, n_cands)

            # Make sure each row is uniformly distributed
            for row in election:
                assert row.min() >= 0
                assert row.max() <= 1

    # Check that utilities are equally distributed from 0 to 1
    n_voters = 10000
    n_cands = 15
    bins = 10
    election = random_utilities(n_voters, n_cands, random_state=rng)
    hist = np.histogram(election.flatten(), bins=bins, range=(0, 1))[0]
    assert_array_less(hist/(n_voters*n_cands), 0.11)


def test_impartial_culture():
    rng = np.random.default_rng(1982)  # Deterministic tests

    for n_voters in (0, 1, 5, 100):
        for n_cands in (0, 1, 2, 3, 7, 100):
            election = impartial_culture(n_voters, n_cands, random_state=rng)

            # Make sure rows are voters and columns are
            # rankings (= same number as candidates)
            assert election.shape == (n_voters, n_cands)

            # Make sure each row is a permutation with no ties
            for row in election:
                # (Empty arrays test equal, which is fine)
                assert_array_equal(np.bincount(row), 1)

    # Check that rankings are equally distributed
    n_voters = 10000
    n_cands = 15
    election = impartial_culture(n_voters, n_cands, random_state=rng)
    for col in election.T:
        assert_array_less(abs(np.bincount(col)/(n_voters/n_cands) - 1),
                          0.15)


def test_invalid():
    with pytest.raises(ValueError):
        impartial_culture(1, 257)


def test_normal_electorate():
    rng = np.random.default_rng(1299)  # Deterministic tests

    n_voters = 10000
    n_cands = 10000
    dims = 2
    corr = 0.3
    disp = 0.2

    voters, cands = normal_electorate(n_voters, n_cands, dims, corr, disp,
                                      random_state=rng)

    # Check dimensions
    assert voters.shape == (n_voters, dims)
    assert cands.shape == (n_cands, dims)

    # Check dispersion
    assert_allclose(np.std(voters[:, 0])*disp,
                    np.std(cands[:, 0]), rtol=0.1)
    assert_allclose(np.std(voters[:, 1])*disp,
                    np.std(cands[:, 1]), rtol=0.1)


def test_normed_dist_utilities():
    voters = [[1, 1],
              [6, 3],
              [1, 7]]
    cands = [[2, 3],
             [5, 1],
             [4, 6]]

    utilities = normed_dist_utilities(voters, cands)

    # Calculated in spreadsheet
    assert_allclose(utilities,
                    [[1.000000000000000, 0.509321562793963, 0.000000000000000],
                     [0.000000000000000, 1.000000000000000, 0.223619005440422],
                     [0.762689671355725, 0.000000000000000, 1.000000000000000]]
                    )


@pytest.mark.parametrize("func", [random_utilities, impartial_culture,
                                  normal_electorate])
def test_random_state(func):
    a = func(5, 5)
    b = func(5, 5)
    c = func(5, 5, random_state=1997)
    d = func(5, 5, random_state=1997)
    e = func(5, 5, random_state=np.random.default_rng(1997))
    f = func(5, 5, random_state=np.random.default_rng(1997))

    # a and b should *not* be equal, nor should a and c
    with pytest.raises(AssertionError):
        assert_array_equal(a, b)

    with pytest.raises(AssertionError):
        assert_array_equal(a, c)

    # All the others should be the same:
    assert_array_equal(c, d)
    assert_array_equal(c, e)
    assert_array_equal(c, f)


@pytest.mark.parametrize("func", [random_utilities, impartial_culture,
                                  normal_electorate])
def test_invalid_random_state(func):
    with pytest.raises(ValueError):
        func(5, 5, random_state='bananas')


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
