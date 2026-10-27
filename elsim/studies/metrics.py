"""Small metric helpers shared by the public studies API."""

from __future__ import annotations

from elsim.methods import approval, utility_winner
from elsim.strategies import approval_optimal, honest_rankings


def approval_at_optimal(utilities, tiebreaker='random'):
    """Optimal approval ballots, then the approval winner."""
    return approval(approval_optimal(utilities), tiebreaker=tiebreaker)


def utility_winner_at(utilities, tiebreaker='random'):
    """Winner with the highest summed utility."""
    return utility_winner(utilities, tiebreaker=tiebreaker)


def normalize_utilities(utilities, mode):
    """
    Normalize each voter's utilities for social-utility comparisons.

    ``'range'`` maps each row to [0, 1]; ``'standard'`` gives mean 0 and
    standard deviation 1; ``None`` or ``'none'`` leaves the values alone.
    """
    if mode is None or mode == 'none':
        return utilities
    if mode == 'range':
        utilities = utilities - utilities.min(axis=1, keepdims=True)
        return utilities / utilities.max(axis=1, keepdims=True)
    if mode == 'standard':
        return ((utilities - utilities.mean(axis=1, keepdims=True))
                / utilities.std(axis=1, keepdims=True))
    raise ValueError(f'unknown utility normalization: {mode!r}')


def rankings_from(utilities):
    """Honest rankings for an array of utilities."""
    return honest_rankings(utilities)
