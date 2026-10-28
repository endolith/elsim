"""Small helpers shared by test modules."""

import numpy as np


def permute_ranked_candidates(election, permutation):
    """Rename candidate IDs in ranked ballots using `permutation`."""
    return np.asarray(permutation)[np.asarray(election)]


def permute_candidate_columns(election, permutation):
    """Reorder candidate columns using `permutation`."""
    return np.asarray(election)[:, np.argsort(permutation)]
