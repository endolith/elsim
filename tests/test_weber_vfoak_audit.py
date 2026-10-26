"""Independent checks of the VFOAK audit, without fitting Weber's column."""

from fractions import Fraction
from itertools import combinations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import qmc

from elsim.strategies import vote_for_or_against_k
from examples.weber_vfoak_audit import (best_k, effectiveness, expected_gain,
                                      for_region_vertices, simulate)


@pytest.mark.parametrize('m,k,gain', [
    (2, 1, Fraction(1, 6)),
    (3, 1, Fraction(7, 24)),
    (4, 1, Fraction(7, 20)),
    (5, 2, Fraction(97, 180)),
    (6, 2, Fraction(79, 126)),
    (10, 4, Fraction(31727, 27720)),
])
def test_exact_integral_against_utilities(m, k, gain):
    """Check rational integration against independently sampled utility cubes."""
    assert expected_gain(m, k) == gain
    u = qmc.Sobol(m, scramble=True, seed=731).random_base2(18)
    centered = np.sort(u - u.mean(axis=1, keepdims=True), axis=1)
    actual = np.maximum(centered[:, -k:].sum(axis=1),
                        -centered[:, :k].sum(axis=1)).mean()
    assert_allclose(actual, float(gain), atol=0.0001, rtol=0)


@pytest.mark.parametrize('m', range(2, 11, 2))
def test_half_reduces_to_vote_for_k(m):
    k = m // 2
    vote_for_k_eff = np.sqrt(3*m*k*(m-k)/(m-1))/(m+1)
    assert_allclose(effectiveness(m, k), vote_for_k_eff)


@pytest.mark.parametrize('m', range(2, 9))
def test_existing_strategy_exhaustively_maximizes_weber_objective(m):
    """Enumerate ALL allowed ballots, not just the two utility-ranked choices."""
    u = np.random.default_rng(984 + m).random((100, m))
    centered = u - u.mean(axis=1, keepdims=True)
    for k in range(1, m//2 + 1):
        allowed = []
        for count in {k, m-k}:
            for subset in combinations(range(m), count):
                ballot = np.zeros(m)
                ballot[list(subset)] = 1
                allowed.append(ballot)
        actual = vote_for_or_against_k(u, k)
        assert set(actual.sum(axis=1)) <= {k, m-k}
        assert_allclose((actual*centered).sum(axis=1),
                        (centered @ np.array(allowed).T).max(axis=1), atol=1e-14)


def test_simulation_with_one_voter():
    """For m=2 and one voter the favorite is always the utility winner."""
    value, se = simulate(2, 1, 100, 1, np.random.default_rng(1))
    assert value == 1
    assert se < 1e-14


@pytest.mark.parametrize('m,k', [(1, 1), (5, 0), (5, 3), (5, 1.5)])
def test_invalid_parameters(m, k):
    with pytest.raises(ValueError):
        expected_gain(m, k)


def test_five_candidate_region_is_not_a_simplex():
    """The page-71 simplex assertion fails: dimension 5, eight vertices."""
    vertices = for_region_vertices(5, 2)
    assert len(vertices) == 8
    assert (Fraction(1), Fraction(1, 3), Fraction(1, 3), 0, 0) in vertices
    assert (Fraction(1), Fraction(1), Fraction(2, 3), Fraction(2, 3), 0) in vertices
    points = np.array(vertices, dtype=float)
    assert np.linalg.matrix_rank(points[1:] - points[0]) == 5


def test_incorrect_six_vertex_centroid_explains_quoted_value():
    """Omitting the two extra vertices recovers precisely the disputed 86.96%."""
    omitted = {(Fraction(1), Fraction(1, 3), Fraction(1, 3), 0, 0),
               (Fraction(1), Fraction(1), Fraction(2, 3), Fraction(2, 3), 0)}
    six = [v for v in for_region_vertices(5, 2) if v not in omitted]
    centroid = [sum(v[i] for v in six)/len(six) for i in range(5)]
    # Direct transcription of page 14's printed h formula at m=5.
    assert centroid == [Fraction(5, 6), Fraction(7, 12), Fraction(1, 3),
                        Fraction(1, 4), Fraction(1, 6)]
    a = sum(centroid[:2])/2
    b = sum(centroid[2:])/3
    assert a == Fraction(17, 24)
    assert b == Fraction(1, 4)
    assert a-b == Fraction(11, 24)
    assert_allclose(float(a-b)*np.sqrt(12*2*3/(5*4)), 0.8696263565463043)


def test_true_region_centroid_by_exact_triangulation():
    """Independent polytope integration, with exact rational determinants."""
    from math import factorial
    from scipy.spatial import ConvexHull

    def determinant(matrix):
        matrix = [list(row) for row in matrix]
        result = Fraction(1)
        for i in range(len(matrix)):
            pivot = next((j for j in range(i, len(matrix)) if matrix[j][i]), None)
            if pivot is None:
                return Fraction(0)
            if pivot != i:
                matrix[i], matrix[pivot] = matrix[pivot], matrix[i]
                result = -result
            value = matrix[i][i]
            result *= value
            for j in range(i+1, len(matrix)):
                weight = matrix[j][i]/value
                matrix[j] = [a-weight*b for a, b in zip(matrix[j], matrix[i])]
        return result

    vertices = for_region_vertices(5, 2)
    # Qhull supplies only facet connectivity; all integration below is rational.
    hull = ConvexHull(np.array(vertices, dtype=float))
    interior = tuple(sum(v[i] for v in vertices)/len(vertices) for i in range(5))
    volume = Fraction()
    moment = [Fraction() for _ in range(5)]
    for facet in hull.simplices:
        points = [vertices[i] for i in facet] + [interior]
        matrix = [[a-b for a, b in zip(p, points[0])] for p in points[1:]]
        weight = abs(determinant(matrix))/factorial(5)
        volume += weight
        for i in range(5):
            moment[i] += weight*sum(p[i] for p in points)/6
    assert volume == Fraction(1, 240)
    centroid = [value/volume for value in moment]
    assert centroid == [Fraction(5, 6), Fraction(67, 108), Fraction(41, 108),
                        Fraction(31, 108), Fraction(1, 6)]
    omega = sum(centroid[:2])/2 - sum(centroid[2:])/3
    assert omega == Fraction(97, 216)
    assert_allclose(float(omega)*np.sqrt(12*2*3/(5*4)), effectiveness(5, 2))


@pytest.mark.parametrize('m,k,eff', [
    (9, 4, 0.8485171511721072),
    (20, 9, 0.8578060289674586),
    (50, 22, 0.8632919416312936),
    (100, 46, 0.8648525407783865),
])
def test_best_k_using_exact_rational_comparisons(m, k, eff):
    """Check the larger cases independently discussed by both Arena instances."""
    assert best_k(m) == k
    assert_allclose(effectiveness(m, k), eff, atol=1e-14, rtol=0)


@pytest.mark.parametrize('m', [3, 4])
def test_small_for_regions_really_are_simplices(m):
    """The paper's geometric shortcut genuinely works in these small cases."""
    assert len(for_region_vertices(m, 1)) == m+1


@pytest.mark.parametrize('m', [4, 6, 8, 10])
def test_half_for_region_is_the_entire_ordered_simplex(m):
    """At k=m/2 both sign choices are equivalent, so the cut disappears."""
    original = [tuple(Fraction(int(i < j)) for i in range(m))
                for j in range(m+1)]
    assert for_region_vertices(m, m//2) == original


@pytest.mark.parametrize('m', [1, 0, -1, 3.5])
def test_best_k_invalid_candidates(m):
    with pytest.raises(ValueError):
        best_k(m)
