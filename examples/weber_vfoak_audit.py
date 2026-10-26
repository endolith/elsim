"""Independent audit of the VFOAK column, not a fit to its quoted values.

Run from the repository root:
    PYTHONPATH=. python examples/weber_vfoak_audit.py
    PYTHONPATH=. python examples/weber_vfoak_audit.py --elections 30000

Exact integration uses uniform order-statistic spacings (a uniform simplex)
and confluent divided differences of the truncated power x_+**(m+1).
See weber_vfoak_analysis.md for the derivation and verification against the
original printed pages supplied by the repository owner.
"""

import argparse
from fractions import Fraction
from math import comb, sqrt

import numpy as np


def expected_gain(m, k):
    """Return exact E[max(sum(top k centered u), -sum(bottom k centered u))]."""
    if not isinstance(m, int) or not isinstance(k, int) or not 1 <= k <= m // 2:
        raise ValueError('Require integers m >= 2 and 1 <= k <= m//2')

    # In ascending order, D = A - B has these coefficients on the utilities.
    coefficients = [Fraction(int(i < k or i >= m - k)) - Fraction(2*k, m)
                    for i in range(m)]
    # Express D in the m+1 uniform order-statistic spacings, including endpoints.
    knots = sorted(sum(coefficients[j:], Fraction()) for j in range(m + 1))
    power = m + 1
    differences = [max(x, Fraction())**power for x in knots]
    for order in range(1, m + 1):
        next_differences = []
        for i in range(m + 1 - order):
            left, right = knots[i], knots[i + order]
            if left == right:
                value = (comb(power, order) * max(left, Fraction())**(power-order))
            else:
                value = (differences[i + 1] - differences[i]) / (right - left)
            next_differences.append(value)
        differences = next_differences
    positive_part = differences[0] / (m + 1)  # E[(A-B)_+]
    # E[A] = E[B] = k(m-k)/(2(m+1)); max(A,B) = B + (A-B)_+.
    return Fraction(k*(m-k), 2*(m+1)) + positive_part


def for_region_vertices(m, k):
    """Exact vertices of the closed, descending-utility 'vote for' region.

    The ordered unit cube is a simplex with vertices (1,...,1,0,...,0).
    Cutting it by A-B >= 0 keeps its nonnegative vertices and introduces
    an intersection on EVERY edge joining strictly opposite signs.
    """
    if not isinstance(m, int) or not isinstance(k, int) or not 1 <= k <= m // 2:
        raise ValueError('Require integers m >= 2 and 1 <= k <= m//2')
    original = [tuple(Fraction(int(i < j)) for i in range(m))
                for j in range(m + 1)]

    def difference(x):
        return sum(x[:k]) + sum(x[-k:]) - Fraction(2*k, m)*sum(x)

    vertices = {x for x in original if difference(x) >= 0}
    for x in original:
        for y in original:
            dx, dy = difference(x), difference(y)
            if dx > 0 > dy:
                weight = -dy / (dx - dy)
                vertices.add(tuple(weight*a + (1-weight)*b for a, b in zip(x, y)))
    return sorted(vertices)


def effectiveness(m, k):
    """Asymptotic election SUE, from an exact rational expected gain."""
    return float(expected_gain(m, k)) * sqrt(12*m / ((m-1)*k*(m-k)))


def best_k(m):
    """Maximize effectiveness using exact squared values, not noisy quadrature.

    The factor 12*m/(m-1) is independent of k. Both gains and their squared
    comparison below are rational, so even close competing k are compared
    without floating-point error. Choose the smaller k on an exact tie.
    """
    if not isinstance(m, int) or m < 2:
        raise ValueError('Require an integer m >= 2')
    return max(range(1, m//2 + 1),
               key=lambda k: expected_gain(m, k)**2 / (k*(m-k)))


def simulate(m, k, elections, voters, rng):
    """Independent full elections; integrate random tie-breaking exactly.

    Subtract each realized society's average candidate utility, an unbiased
    variance reduction relative to subtracting voters/2. The ratio is of
    expectations, NOT the expectation of per-election efficiency ratios.
    """
    numerator, denominator = [], []
    for start in range(0, elections, 100):
        size = min(100, elections - start)
        u = rng.random((size, voters, m))
        order = np.argsort(u, axis=2)
        centered = u - u.mean(axis=2, keepdims=True)
        ranked = np.take_along_axis(centered, order, axis=2)
        against = -ranked[:, :, :k].sum(axis=2) > ranked[:, :, -k:].sum(axis=2)
        # Approval-equivalent ballots: top k OR top m-k.
        counts = np.where(against, m-k, k)
        ranked_ballots = np.arange(m)[None, None, :] >= (m-counts[:, :, None])
        ballots = np.take_along_axis(ranked_ballots, np.argsort(order, axis=2), axis=2)
        scores = ballots.sum(axis=1)
        tied = scores == scores.max(axis=1, keepdims=True)
        total_u = u.sum(axis=1)
        baseline = total_u.mean(axis=1)
        numerator.extend((total_u*tied).sum(axis=1)/tied.sum(axis=1) - baseline)
        denominator.extend(total_u.max(axis=1) - baseline)
    numerator, denominator = np.array(numerator), np.array(denominator)
    ratio = numerator.mean() / denominator.mean()
    se = (numerator-ratio*denominator).std(ddof=1) / sqrt(elections) / denominator.mean()
    return ratio, se


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--elections', type=int, default=0)
    parser.add_argument('--voters', type=int, default=1001)
    parser.add_argument('--seed', type=int, default=20261025)
    args = parser.parse_args()
    if args.elections < 0 or args.elections == 1 or args.voters < 1:
        parser.error('Use 0 or at least 2 elections, and at least 1 voter')
    rng = np.random.default_rng(args.seed)
    print('m   k*   exact gain   exact SUE   quoted SUE' +
          ('   election SUE (95% MC interval)' if args.elections else ''))
    for m, quoted in [(2, 81.65), (3, 87.50), (4, 80.83),
                      (5, 86.96), (6, 86.25), (10, 88.09)]:
        k = best_k(m)
        row = (f'{m:2}   {k:2}   {str(expected_gain(m,k)):>10}   '
               f'{100*effectiveness(m,k):9.4f}   {quoted:10.2f}')
        if args.elections:
            value, se = simulate(m, k, args.elections, args.voters, rng)
            row += f'   {100*value:.3f} +/- {196*se:.3f}'
        print(row)


if __name__ == '__main__':
    main()
