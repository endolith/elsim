"""
Tools for Monte Carlo election studies and paper-style reproduction scripts.

This subpackage supports declarative spatial-model sweeps (see
:mod:`elsim.studies.spatial_normal`), parameter expansion for scenario grids,
serial batching via :func:`run_batched`, and small tallies shared by several
examples (Condorcet agreement, social-utility totals).

The ``elections`` / ``strategies`` / ``methods`` modules remain the core model;
``studies`` orchestrates repeated draws and aggregation.
"""

from .backends import SerialBackend
from .condorcet_metrics import approval_at_optimal, tally_condorcet_agreement
from .parameters import expand_product, expand_rows, expand_zip
from .runner import merge_counters, run_batched
from .social_utility import (
    random_society_utility_updates,
    ranked_rated_utility_updates,
    spatial_random_reference_utility_updates,
)
from .spatial_normal import accumulate_spatial_condorcet_by_ncands, accumulate_spatial_sue_by_ncands

__all__ = [
    "SerialBackend",
    "approval_at_optimal",
    "expand_product",
    "expand_rows",
    "expand_zip",
    "merge_counters",
    "run_batched",
    "tally_condorcet_agreement",
    "accumulate_spatial_condorcet_by_ncands",
    "accumulate_spatial_sue_by_ncands",
    "spatial_random_reference_utility_updates",
    "random_society_utility_updates",
    "ranked_rated_utility_updates",
]
