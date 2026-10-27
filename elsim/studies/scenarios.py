"""
Declare Monte Carlo scenarios as plain dictionaries.

Two shapes cover the example scripts:

* :func:`grid` expands any iterable parameter and keeps scalars fixed,
  giving a one-axis sweep or a full Cartesian product.
* :func:`conditions` takes an explicit list of rows, for paper tables whose
  rows are hand-picked parameter combinations rather than a product.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from itertools import product
from typing import Any


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)):
        return (value,)
    if isinstance(value, Mapping):
        raise TypeError(
            'grid() parameters must be scalars or iterables, not mappings')
    if isinstance(value, Iterable):
        return tuple(value)
    return (value,)


def grid(**axes: Any) -> list[dict[str, Any]]:
    """
    Expand parameter axes into scenario dictionaries.

    Scalars stay fixed; iterables are expanded.  One iterable gives a
    one-axis sweep; several give their Cartesian product.

    Examples
    --------
    >>> grid(n_cands=[2, 3], n_voters=25)
    [{'n_cands': 2, 'n_voters': 25}, {'n_cands': 3, 'n_voters': 25}]

    >>> grid(n_voters=[3, 5], n_cands=[3, 4])  # doctest: +NORMALIZE_WHITESPACE
    [{'n_voters': 3, 'n_cands': 3}, {'n_voters': 3, 'n_cands': 4},
     {'n_voters': 5, 'n_cands': 3}, {'n_voters': 5, 'n_cands': 4}]
    """
    if not axes:
        return [{}]
    keys = list(axes)
    values = [_as_tuple(axes[key]) for key in keys]
    return [dict(zip(keys, combo, strict=True))
            for combo in product(*values)]


def conditions(
    rows: Sequence[Sequence[Any] | Mapping[str, Any]],
    keys: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Turn explicit rows into scenario dictionaries.

    Rows may already be mappings, or sequences paired with ``keys``.  Use
    this for hand-picked paper conditions such as Merrill Table 4's
    ``(disp, corr, D)`` columns.

    Examples
    --------
    >>> conditions([(1.0, 0.5, 2), (0.5, 0.0, 2)], ('disp', 'corr', 'D'))
    [{'disp': 1.0, 'corr': 0.5, 'D': 2}, {'disp': 0.5, 'corr': 0.0, 'D': 2}]
    """
    out: list[dict[str, Any]] = []
    keys_t = tuple(keys) if keys is not None else None
    for index, row in enumerate(rows):
        if isinstance(row, Mapping):
            if keys_t is not None:
                raise ValueError(
                    'keys were given but row is already a mapping')
            out.append(dict(row))
            continue
        if keys_t is None:
            raise ValueError('keys are required when rows are sequences')
        row_t = tuple(row)
        if len(row_t) != len(keys_t):
            raise ValueError(
                f'row {index} has {len(row_t)} values but {len(keys_t)} '
                'keys were given')
        out.append(dict(zip(keys_t, row_t, strict=True)))
    return out
