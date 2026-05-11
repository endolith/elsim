"""
Expand simulation parameters into explicit scenario dictionaries.

Issue `#10 <https://github.com/endolith/elsim/issues/10>`_ called out three
common shapes:

* **Cartesian product** — every combination of voter counts, candidate counts,
  methods, etc.  Use :func:`expand_product`.
* **Zipped columns** — parallel lists of the same length (e.g. sweep ``n_voters``
  and ``n_cands`` together).  Use :func:`expand_zip`.
* **Explicit rows** — a small table of tuples such as Merrill 1984 Table 2
  ``(disp, corr, D)`` that is *not* a full product.  Use :func:`expand_rows` or
  pass your own sequence of mappings to your driver loop.

Strings are treated as scalars (not iterated character-wise).
"""

from __future__ import annotations

from itertools import product
from typing import Any, Iterable, Mapping, Sequence, Union

Scalar = Any
ScalarOrIterable = Union[Scalar, Iterable[Scalar]]


def _as_tuple(x: ScalarOrIterable) -> tuple[Scalar, ...]:
    if isinstance(x, (str, bytes)):
        return (x,)
    if isinstance(x, Mapping):
        raise TypeError("Mappings are not treated as iterables of scenarios; "
                        "pass keys to expand_product or use expand_rows.")
    if isinstance(x, Iterable):
        return tuple(x)
    return (x,)


def expand_product(**params: ScalarOrIterable) -> list[dict[str, Any]]:
    """
    Cartesian product of parameter values.

    Each keyword argument is either a single value or an iterable of values.
    The return value is a list of dicts, one per combination, in deterministic
    order (same as :func:`itertools.product`).

    Examples
    --------
    >>> expand_product(n_voters=[10, 20], n_cands=3)
    [{'n_voters': 10, 'n_cands': 3}, {'n_voters': 20, 'n_cands': 3}]
    """
    if not params:
        return [{}]
    keys = list(params)
    value_lists = [_as_tuple(params[k]) for k in keys]
    return [dict(zip(keys, combo)) for combo in product(*value_lists)]


def expand_zip(**params: Iterable) -> list[dict[str, Any]]:
    """
    Zip parallel parameter columns into scenario dicts.

    All iterables must have the same length.

    Parameters
    ----------
    **params
        Each value must be an iterable of scenario values for that key.

    Examples
    --------
    >>> expand_zip(n_voters=[100, 200], n_cands=[3, 5])
    [{'n_voters': 100, 'n_cands': 3}, {'n_voters': 200, 'n_cands': 5}]
    """
    if not params:
        return []
    keys = list(params)
    columns = [list(params[k]) for k in keys]
    lengths = {len(c) for c in columns}
    if len(lengths) > 1:
        raise ValueError(
            "expand_zip: all parameter lists must have the same length; "
            f"got {dict(zip(keys, map(len, columns)))}"
        )
    rows = zip(*columns)
    return [dict(zip(keys, row)) for row in rows]


def expand_rows(rows: Sequence[Sequence[Any]], keys: Sequence[str]) -> list[dict[str, Any]]:
    """
    Turn fixed scenario rows into dicts.

    Use this for tables like Merrill (1984) Table 2 where each row is a
    deliberate ``(disp, corr, D)`` triple rather than a combination from a grid.

    Parameters
    ----------
    rows : sequence of row sequences
        Each inner sequence must have ``len(keys)`` entries.
    keys : sequence of str
        Names for each column.

    Examples
    --------
    >>> expand_rows([(1.0, 0.5, 2), (0.5, 0.0, 4)], ('disp', 'corr', 'D'))
    [{'disp': 1.0, 'corr': 0.5, 'D': 2}, {'disp': 0.5, 'corr': 0.0, 'D': 4}]
    """
    keys_t = tuple(keys)
    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        row_t = tuple(row)
        if len(row_t) != len(keys_t):
            raise ValueError(
                f"expand_rows: row {i} has length {len(row_t)} but {len(keys_t)} keys were given"
            )
        out.append(dict(zip(keys_t, row_t)))
    return out
