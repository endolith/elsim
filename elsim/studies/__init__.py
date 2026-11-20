"""
Declarative Monte Carlo studies.

Scenarios are plain dictionaries (see :func:`grid` and :func:`conditions`),
electorate models draw them (`spatial_normal`, `random_society`,
`impartial_culture`), and :class:`Study` runs a metric with optional
parallelism and candidate-set grouping.

The core ``elections`` / ``strategies`` / ``methods`` modules remain the
building blocks; ``studies`` only orchestrates repeated draws and aggregation.
"""

from .metrics import (approval_at_optimal, normalize_utilities,
                      utility_winner_at)
from .models import (Election, ImpartialCulture, RandomSociety,
                     SpatialNormal, impartial_culture, random_society,
                     spatial_normal)
from .scenarios import conditions, grid
from .study import Study, StudyResult, run

__all__ = [
    'Election',
    'ImpartialCulture',
    'RandomSociety',
    'SpatialNormal',
    'Study',
    'StudyResult',
    'approval_at_optimal',
    'conditions',
    'grid',
    'impartial_culture',
    'normalize_utilities',
    'random_society',
    'run',
    'spatial_normal',
    'utility_winner_at',
]
