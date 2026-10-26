"""
Where Weber's vote-for-or-against-k formula goes wrong.

Weber, Robert J. (1978). "Comparison of Public Choice Systems".
Cowles Foundation Discussion Papers. Cowles Foundation for Research in
Economics. No. 498. https://cowles.yale.edu/publications/cfdp/cfdp-498

Pages 14-16 derive the effectiveness of "vote for or against k"; page 19 prints
the resulting column, whose values (86.96%, 86.25%, 88.09%, 92.25% for
m = 5, 6, 10, inf) are too high from m = 5 on.  This example shows what goes
wrong, and produces the correct numbers exactly, with no simulation.

The mechanism
-------------
For an m-candidate election let S be the set of candidates of a type in T_1, so
u_t(c) = a_1 if c is in S and b_1 if not; symmetrically for T_2, with a_2, b_2.
Weber's page-15 identity

    sum_t p_t u_t(c)[u_t(c) - u_t(d)] = (1 / (2 C(m,k))) * C(m-2,k-1) * 2 w^2,
    where w = a_1 - b_1 = a_2 - b_2,

is correct, and it reduces the whole problem to one number, w.  The four
numbers a_1, b_1, a_2, b_2 come from the centroid h of the region

    H = {u_1 >= u_2 >= ... >= u_m,  (u_{k+1} + ... + u_{m-k})/(m - 2k) <= ubar},

the set of voters for whom "vote for the top k" scores at least as well as
"vote for the top m-k".  (a_1 is the average of the first k components of h,
b_1 the average of the last m-k, etc.)

The error is one sentence on page 14: "This region is a simplex, with extreme
points ...".  H is a polytope, and for m >= 5 it is not a simplex -- it has far
more than m + 1 extreme points.  A simplex's centroid is the average of its
vertices, but a general polytope's is not, so h (and hence w, and hence the
published formula) is wrong.  The cases where the paper's formula is right are
exactly the cases where H *is* a simplex:

* m <= 4, where H has m + 1 extreme points (this is where the paper worked the
  system out by hand), and
* k = m/2, where the inequality is vacuous and H is the whole sorted simplex.

`region_centroid` below computes the centroid of H exactly (vertex enumeration,
then a decomposition into pyramids), so the correct effectiveness comes out as
an exact rational number -- 75.0391% for (m, k) = (5, 1), for instance, in
agreement with the quadrature in weber_1977_expressions.central_gap.
"""
import math
import numpy as np
from fractions import Fraction
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection
from tabulate import tabulate
from weber_1977_expressions import (eff_vote_for_or_against_k,
                                    eff_vote_for_or_against_k_paper,
                                    best_vote_for_or_against_k)


def region_constraints(m, k):
    """
    Halfspaces ``A v <= b`` describing H, in the sorted coordinates v.
    """
    A, b = [], []
    for i in range(m - 1):          # v_i >= v_{i+1}
        row = np.zeros(m)
        row[i], row[i+1] = -1, 1
        A.append(row)
        b.append(0)
    row = np.zeros(m)               # v_1 <= 1
    row[0] = 1
    A.append(row)
    b.append(1)
    row = np.zeros(m)               # v_m >= 0
    row[m-1] = -1
    A.append(row)
    b.append(0)
    # 2k * (central sum) <= (m - 2k) * (top sum + bottom sum), i.e. central
    # mean <= ubar.  Vacuous when k = m/2.
    row = np.zeros(m)
    row[k:m-k] += 2*k
    row[:k] -= (m - 2*k)
    row[m-k:] -= (m - 2*k)
    A.append(row)
    b.append(0)
    A, b = np.array(A, float), np.array(b, float)
    keep = np.abs(A).sum(axis=1) > 0
    return A[keep], b[keep]


def region_vertices(m, k):
    """
    The extreme points of H, sorted lexicographically.
    """
    A, b = region_constraints(m, k)
    norms = np.linalg.norm(A, axis=1)
    # Chebyshev centre: a point strictly inside, which qhull requires
    res = linprog(np.r_[np.zeros(m), -1], A_ub=np.hstack([A, norms[:, None]]),
                  b_ub=b, bounds=[(None, None)]*m + [(0, None)], method='highs')
    hull = HalfspaceIntersection(np.hstack([A, -b[:, None]]), res.x[:m])
    vertices = hull.intersections
    return vertices[np.lexsort(vertices.T[::-1])]


def _facet_measure(points):
    """(m-1)-dimensional volume of a facet with m vertices."""
    d = points[1:] - points[0]
    return math.sqrt(max(np.linalg.det(d @ d.T), 0.0))/math.factorial(len(d))


def region_centroid(m, k):
    """
    Exact centroid of H, as a list of Fractions.

    H is convex and contains the origin as a vertex, so it is the union of the
    pyramids over its facets that do not contain the origin.  The centroid of a
    pyramid with apex p, base centroid c and m dimensions is (p + m c)/(m + 1).
    """
    vertices = region_vertices(m, k)
    hull = ConvexHull(vertices)
    apex = np.zeros(m)
    total = 0.0
    acc = np.zeros(m)
    for facet, equation in zip(hull.simplices, hull.equations):
        points = vertices[facet]
        height = abs(equation[m])       # qhull's normals are unit vectors
        volume = _facet_measure(points)*height/m
        if volume < 1e-12:
            continue                    # facet contains the apex
        total += volume
        acc += volume*(apex + m*points.mean(axis=0))/(m + 1)
    return [Fraction(x).limit_denominator(10**6) for x in acc/total]


def paper_a1(m, k):
    """a_1 as tabulated on page 14 of the paper (same for even and odd m)."""
    return (2*m - 1.5*(k - 1))/(2*(m + 1))


def paper_b1(m, k):
    """b_1 as tabulated on page 14 of the paper."""
    numerator = 3*m*m + 6*m + (-1 if m % 2 else 0) + 6*k*k - 6*k - 8*m*k
    return numerator/(8*(m + 1)*(m - k))


def exact_effectiveness(m, k):
    """
    Effectiveness from the exact centroid of H.
    """
    h = region_centroid(m, k)
    a1 = float(np.mean(h[:k]))
    b1 = float(np.mean(h[k:]))
    return (a1 - b1)*math.sqrt(12*k*(m - k)/(m*(m - 1)))


if __name__ == '__main__':
    # The exact centroid must reproduce the quadrature in weber_1977_expressions
    for m, k in ((3, 1), (4, 1), (5, 1), (5, 2), (6, 1), (6, 2), (7, 2)):
        assert abs(exact_effectiveness(m, k)
                   - eff_vote_for_or_against_k(m, k)) < 1e-4, (m, k)

    rows = []
    for m in range(2, 10):
        for k in range(1, m//2 + 1):
            vertices = region_vertices(m, k)
            h = region_centroid(m, k)
            a1, b1 = float(np.mean(h[:k])), float(np.mean(h[k:]))
            rows.append([m, k, len(vertices), m + 1,
                         f'{len(vertices) == m + 1}',
                         f'{paper_a1(m, k):.5f}', f'{a1:.5f}',
                         f'{paper_b1(m, k):.5f}', f'{b1:.5f}',
                         f'{eff_vote_for_or_against_k_paper(m, k):.2%}',
                         f'{exact_effectiveness(m, k):.2%}'])

    print('The region H on p. 14 of the paper, versus its claim that H is a')
    print('simplex with m + 1 extreme points:\n')
    print(tabulate(rows, headers=('m', 'k', 'extreme points', 'm + 1',
                                  'simplex?', 'a1 paper', 'a1 exact',
                                  'b1 paper', 'b1 exact',
                                  'published', 'correct'),
                   tablefmt='pipe', disable_numparse=True))

    print('\nThe centroid of H, for the two smallest failing cases:\n')
    for m, k in ((5, 1), (6, 2)):
        h = region_centroid(m, k)
        print(f'  m = {m}, k = {k}')
        print(f'    exact h = {[str(x) for x in h]}')
        print(f'    exact a1, b1 = {float(np.mean(h[:k])):.6f}, '
              f'{float(np.mean(h[k:])):.6f}  ->  w = '
              f'{float(np.mean(h[:k])) - float(np.mean(h[k:])):.6f}')
        print(f'    paper a1, b1 = {paper_a1(m, k):.6f}, {paper_b1(m, k):.6f}'
              f'  ->  w = {paper_a1(m, k) - paper_b1(m, k):.6f}')
        print()

    m, k = 5, 1
    listed = [(0, 0, 0, 0, 0), (1, 0, 0, 0, 0), (1, 1, 0, 0, 0),
              (1, 1, .5, 0, 0), (1, .5, .5, .5, 0), (1, 1, 1, 1, 1)]
    h = np.mean(listed, axis=0)
    print(f'The {len(listed)} points the paper lists for m = 5, k = 1 average to')
    print(f'  {np.round(h, 5)}, i.e. a1 = {h[0]:.5f} and b1 = {h[1:].mean():.5f} --')
    print(f'exactly the a1 and b1 it tabulates ({paper_a1(m, k):.5f}, '
          f'{paper_b1(m, k):.5f}).  But H has these')
    print(f'{len(region_vertices(m, k))} extreme points:\n')
    for v in np.abs(region_vertices(m, k)):
        missing = '' if any(np.allclose(v, p) for p in listed) else '   <- missing'
        print(f'    {np.round(v, 4)}{missing}')
    print('\nThe two missing points pull the centroid of the middle components')
    print('down, so the paper\'s b1 is too small and w -- and the published')
    print('effectiveness -- too large.  For m = 3 and m = 4 nothing is missing,')
    print('which is why the published formula is exact there.')
    print()
    print('For k = m/2 the inequality is vacuous and H is the whole sorted')
    print('simplex, so its centroid is the average of the m + 1 points')
    print('(1,...,1), (1,...,1,0), ... and again there is no error.  The')
    print('published a1 and b1 are individually wrong there (e.g. 0.64286 and')
    print('0.21429 instead of 0.71429 and 0.28571 for m = 6, k = 3), but only')
    print('their difference enters the formula, and that is right.')
