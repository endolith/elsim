"""ASV timings for ``elsim.methods``."""

import numpy as np

from elsim.elections import impartial_culture, random_utilities
from elsim.methods import (
    approval,
    black,
    borda,
    combined_approval,
    condorcet,
    condorcet_from_matrix,
    coombs,
    fptp,
    irv,
    matrix_from_scores,
    ranked_election_to_matrix,
    runoff,
    score,
    sntv,
    star,
    utility_winner,
)
from elsim.strategies import approval_optimal, honest_normed_scores

N_VOTERS = 800
N_CANDS = 7


class RankedBallotMethods:
    def setup_cache(self):
        election = impartial_culture(N_VOTERS, N_CANDS, random_state=42)
        return np.asarray(election)

    def time_black(self, election):
        black(election)

    def time_borda(self, election):
        borda(election)

    def time_fptp(self, election):
        fptp(election)

    def time_sntv(self, election):
        sntv(election, n=2)

    def time_runoff(self, election):
        runoff(election)

    def time_irv(self, election):
        irv(election)

    def time_coombs(self, election):
        coombs(election)


class ScoredBallotMethods:
    def setup_cache(self):
        utilities = random_utilities(N_VOTERS, N_CANDS, random_state=42)
        election = honest_normed_scores(utilities)
        return np.asarray(election)

    def time_score(self, election):
        score(election)

    def time_star(self, election):
        star(election)

    def time_matrix_from_scores(self, election):
        matrix_from_scores(election)


class ApprovalBallotMethods:
    def setup_cache(self):
        utilities = random_utilities(N_VOTERS, N_CANDS, random_state=42)
        election = approval_optimal(utilities)
        return np.asarray(election)

    def time_approval(self, election):
        approval(election)


class CombinedApprovalBallots:
    def setup_cache(self):
        rng = np.random.default_rng(42)
        return rng.integers(-1, 2, size=(N_VOTERS, N_CANDS))

    def time_combined_approval(self, election):
        combined_approval(election)


class UtilityWinnerBallots:
    def setup_cache(self):
        return random_utilities(N_VOTERS, N_CANDS, random_state=42)

    def time_utility_winner(self, utilities):
        utility_winner(utilities)


class CondorcetSuite:
    def setup_cache(self):
        election = np.asarray(
            impartial_culture(N_VOTERS, N_CANDS, random_state=42)
        )
        matrix = ranked_election_to_matrix(election)
        return {"election": election, "matrix": matrix}

    def time_ranked_election_to_matrix(self, data):
        ranked_election_to_matrix(data["election"])

    def time_condorcet(self, data):
        condorcet(data["election"])

    def time_condorcet_from_matrix(self, data):
        condorcet_from_matrix(data["matrix"])
