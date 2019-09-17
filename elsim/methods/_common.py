"""
The first solutions I tried, such as np.unique(return_counts=True), were very
slow, so I made this numba solution, but then discovered np.add.at, which is
faster than the previous solutions, but still slower than numba.  So I'm using
numba as a "soft" dependency, falling back on numpy if not installed.
"""
from warnings import warn
import numpy as np

try:
    from numba import njit
    numba_enabled = True
except ImportError:
    warn('Numba not installed, Condorcet code will run slower')
    numba_enabled = False

    def njit(*args, **kwargs):
        """
        Do-nothing dummy decorator for when numba not installed
        """
        def decorator(func):
            return func
        return decorator

if numba_enabled:
    @njit(cache=True, nogil=True)
    def _tally_pairs(pairs, tally):
        """
        Takes a 3D array of pairs and a 2D tallying array and modifies the
        tallying array in-place to tally the pairs
        """
        for i in range(pairs.shape[0]):
            for j in range(pairs.shape[1]):
                pair = pairs[i][j]
                tally[pair[0], pair[1]] += 1
else:
    def _tally_pairs(pairs, tally):
        """
        Takes a 3D array of pairs and a 2D tallying array and modifies the
        tallying array in-place to tally the pairs
        """
        np.add.at(tally, tuple(pairs.T), 1)


# https://stackoverflow.com/a/6294205/125507
def _all_indices(iterable, value):
    """
    Return all indices of `iterable` that match `value`.
    """
    return [i for i, x in enumerate(iterable) if x == value]
