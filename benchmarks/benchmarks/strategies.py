"""ASV timings for ``elsim.strategies``."""

from elsim.elections import random_utilities
from elsim.strategies import (
    approval_optimal,
    honest_normed_scores,
    honest_rankings,
    vote_for_k,
)

N_VOTERS = 800
N_CANDS = 7


class StrategiesSuite:
    def setup_cache(self):
        return random_utilities(N_VOTERS, N_CANDS, random_state=42)

    def time_honest_rankings(self, utilities):
        honest_rankings(utilities)

    def time_honest_normed_scores(self, utilities):
        honest_normed_scores(utilities)

    def time_approval_optimal(self, utilities):
        approval_optimal(utilities)

    def time_vote_for_k(self, utilities):
        vote_for_k(utilities, 3)
