"""ASV timings for ``elsim.elections``."""

import numpy as np

from elsim.elections import (
    impartial_culture,
    normal_electorate,
    normed_dist_utilities,
    random_utilities,
)

N_VOTERS = 800
N_CANDS = 7


class ElectionsSuite:
    """Fixed electorate size; spatial model uses two dimensions by default."""

    def setup_cache(self):
        rng = np.random.default_rng(42)
        spatial = normal_electorate(N_VOTERS, N_CANDS, random_state=rng)
        return {
            "rng": np.random.default_rng(43),
            "spatial": spatial,
        }

    def time_random_utilities(self, cache):
        random_utilities(N_VOTERS, N_CANDS, random_state=cache["rng"])

    def time_impartial_culture(self, cache):
        impartial_culture(N_VOTERS, N_CANDS, random_state=cache["rng"])

    def time_normal_electorate(self, cache):
        normal_electorate(N_VOTERS, N_CANDS, random_state=cache["rng"])

    def time_normed_dist_utilities(self, cache):
        voters, cands = cache["spatial"]
        normed_dist_utilities(voters, cands)
