"""
Tools for Monte Carlo election studies and paper-style reproduction scripts.

This subpackage addresses the design goals in
https://github.com/endolith/elsim/issues/10: reusable parameter expansion
(Cartesian product vs. zipped columns vs. explicit scenario rows), batched trial
execution with a swappable backend (serial or Joblib), including repeating one
worker or mapping a list of independent zero-argument callables (for example
``functools.partial`` batch jobs), and small tallies shared by several examples.

The ``elections`` / ``strategies`` / ``methods`` modules remain the core model;
``studies`` only orchestrates repeated draws and aggregation.
"""

from .backends import JoblibBackend, SerialBackend
from .condorcet_metrics import merrill_1984_comparison_methods, tally_condorcet_agreement
from .parameters import expand_product, expand_rows, expand_zip
from .runner import merge_counters, run_batched

__all__ = [
    "JoblibBackend",
    "SerialBackend",
    "expand_product",
    "expand_rows",
    "expand_zip",
    "merge_counters",
    "run_batched",
    "merrill_1984_comparison_methods",
    "tally_condorcet_agreement",
]
