"""
Implements various strategies.

These take collections of utilities as inputs and return collections of
ballots that voters cast for a voting method.
"""
from .strategies import (honest_rankings, honest_normed_scores,
                         approval_optimal, vote_for_k)
