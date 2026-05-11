"""
Implements various strategies.

These take collections of utilities as inputs and return collections of
ballots that voters cast for a voting method.
"""
from elsim.strategies._core import (
    approval_optimal,
    honest_normed_scores,
    honest_rankings,
    vote_for_k,
)

__all__ = [
    'approval_optimal',
    'honest_normed_scores',
    'honest_rankings',
    'vote_for_k',
]
