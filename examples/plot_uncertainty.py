"""
Uncertainty displays for Monte Carlo election simulations.

Binomial counts use two-sided exact (Clopper–Pearson) intervals via
`scipy.stats.binomtest`, i.e. inversion of the binomial test — a standard
frequentist CI for a binomial proportion.

Social utility efficiency is computed as a ratio of correlated per-election
totals: \\hat\\theta = \\sum_i W_i / \\sum_i Z_i with paired (W_i, Z_i).
Uncertainty for that ratio uses a paired bootstrap percentile interval on the
same statistic (resampling simulation indices), which targets the sampling
distribution of the plug-in estimator actually plotted.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest

CONFIDENCE_LEVEL = 0.95


def binomial_proportion_ci_errors_percent(
    k: np.ndarray | list | int,
    n: np.ndarray | list | int,
) -> tuple[np.ndarray, np.ndarray] | tuple[float, float]:
    """
    Vertical errors for matplotlib errorbar when y is the sample proportion * 100.

    Parameters
    ----------
    k : number of successes
    n : number of trials
    """
    k_arr = np.atleast_1d(np.asarray(k, dtype=np.int64))
    n_arr = np.atleast_1d(np.asarray(n, dtype=np.int64))
    lo_e = np.empty(k_arr.shape, dtype=float)
    hi_e = np.empty(k_arr.shape, dtype=float)
    for i in np.ndindex(k_arr.shape):
        ki, ni = int(k_arr[i]), int(n_arr[i])
        if ni <= 0:
            lo_e[i] = hi_e[i] = np.nan
            continue
        p_hat = ki / ni
        ci = binomtest(ki, ni).proportion_ci(
            confidence_level=CONFIDENCE_LEVEL,
            method='exact',
        )
        lo_e[i] = (p_hat - ci.low) * 100
        hi_e[i] = (ci.high - p_hat) * 100
    if lo_e.size == 1:
        return float(lo_e.flat[0]), float(hi_e.flat[0])
    return lo_e, hi_e


def sue_ratio_point_and_errors_percent(
    W: np.ndarray,
    Z: np.ndarray,
    *,
    n_resamples: int = 3999,
    rng: np.random.Generator | int | None = None,
) -> tuple[float, float, float]:
    """
    Point estimate y_hat = 100 * sum(W) / sum(Z) and asymmetric errors
    (lower, upper) in percentage points for matplotlib errorbar.

    Uses a percentile bootstrap on the same ratio with paired resampling of
    simulation indices. Rows with invalid pairs are omitted.
    """
    rng = np.random.default_rng(rng)
    W = np.asarray(W, dtype=float)
    Z = np.asarray(Z, dtype=float)
    valid = np.isfinite(W) & np.isfinite(Z)
    Wv = W[valid]
    Zv = Z[valid]
    if Wv.size < 2:
        return np.nan, np.nan, np.nan
    den = np.sum(Zv)
    if den == 0 or not np.isfinite(den):
        return np.nan, np.nan, np.nan
    point = float(np.sum(Wv) / den * 100)
    n = Wv.size
    idx = rng.integers(0, n, size=(n_resamples, n))
    Wb = Wv[idx]
    Zb = Zv[idx]
    sum_w = np.sum(Wb, axis=1)
    sum_z = np.sum(Zb, axis=1)
    ok = np.isfinite(sum_w) & np.isfinite(sum_z) & (sum_z != 0)
    ratios = np.full(n_resamples, np.nan, dtype=float)
    ratios[ok] = (sum_w[ok] / sum_z[ok]) * 100
    ratios = ratios[np.isfinite(ratios)]
    if ratios.size < max(50, n_resamples // 20):
        return point, np.nan, np.nan
    alpha = (1.0 - CONFIDENCE_LEVEL) / 2.0
    lo, hi = np.quantile(ratios, (alpha, 1.0 - alpha))
    return point, point - lo, hi - point


def sue_ratio_curve_points_and_errors(
    W_rows: np.ndarray,
    Z_rows: np.ndarray,
    *,
    n_resamples: int = 3999,
    rng: np.random.Generator | int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Point estimates and asymmetric errors along candidate-count index.

    Parameters
    ----------
    W_rows, Z_rows : shape (n_candidate_counts, n_elections)
        Paired per-election adjusted totals for one method (W) and baseline (Z).
    """
    rng = np.random.default_rng(rng)
    W_rows = np.asarray(W_rows, dtype=float)
    Z_rows = np.asarray(Z_rows, dtype=float)
    n_j = W_rows.shape[0]
    y = np.empty(n_j)
    lo_e = np.empty(n_j)
    hi_e = np.empty(n_j)
    for j in range(n_j):
        pt, le, ue = sue_ratio_point_and_errors_percent(
            W_rows[j],
            Z_rows[j],
            n_resamples=n_resamples,
            rng=rng,
        )
        y[j] = pt
        lo_e[j] = le
        hi_e[j] = ue
    return y, lo_e, hi_e
