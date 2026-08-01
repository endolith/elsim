"""Tests for the pure geometry helpers used by collapse examples."""

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from examples.collapse_2d_shared import (
    count_wins,
    create_frame_scaffold,
    get_theme,
    prepare_palette_and_labels,
    setup_scatter_axis_sigma,
    sort_candidates_bell_curve,
    voronoi_plot_2d_axes,
)


def test_sort_candidates_bell_curve_orders_hemispheres_around_origin():
    """The helper should order candidates outward-in, center, and in-out."""
    candidates = np.array([
        [-2.0, 0.0],
        [1.0, 0.0],
        [0.0, 0.0],
        [-1.0, 0.0],
        [2.0, 0.0],
    ])

    ordered = sort_candidates_bell_curve(candidates)

    np.testing.assert_array_equal(
        ordered,
        [
            [-2.0, 0.0],
            [-1.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ],
    )


def test_two_point_voronoi_handles_same_x_coordinates():
    """A vertical perpendicular bisector must not divide by zero."""
    fig, axis = plt.subplots()
    axis.set_xlim(-2, 2)
    axis.set_ylim(-2, 2)

    voronoi_plot_2d_axes(
        axis,
        np.array([[0.0, -1.0], [0.0, 1.0]]),
    )

    assert len(axis.lines) == 1
    plt.close(fig)


def test_count_wins_counts_only_strict_pairwise_wins():
    """Pairwise ties should not be counted as wins for either candidate."""
    matrix = np.array([
        [0, 3, 2],
        [2, 0, 3],
        [3, 2, 0],
    ])

    assert count_wins(matrix) == [1, 1, 1]


def test_palette_and_theme_helpers_return_rendering_configuration():
    """Palette labels and theme colors should be ready for shared rendering."""
    colors, labels = prepare_palette_and_labels('Bold_10', 3, True)

    assert len(colors) == 3
    assert labels == 'ABC'
    assert get_theme(True)[-1] == '0.15'
    assert get_theme(False)[-1] == '0.88'


def test_scatter_axis_uses_only_visible_sigma_ticks():
    """Scatter axes should place ticks inside the ±1.5σ plot limits."""
    fig, axis = plt.subplots()
    voters = np.array([[-1.0, 0.0], [1.0, 0.0]])
    setup_scatter_axis_sigma(axis, voters)

    sigma = np.std(voters)
    np.testing.assert_array_equal(axis.get_xticks(), [-sigma, 0.0, sigma])
    plt.close(fig)


def test_frame_scaffold_builds_common_panels_and_theme():
    """The scaffold should construct all shared panels for method renderers."""
    voters = np.array([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    candidates = np.array([[-1.0, 0.0], [0.0, 0.5], [1.0, 0.0]])
    colors, labels = prepare_palette_and_labels('Bold_10', 3, True)
    fig, axes, _, theme = create_frame_scaffold(
        voters,
        candidates,
        np.array([0, 1, 2]),
        np.array([50.0, 60.0, 50.0]),
        [1, 1, 1],
        colors,
        labels,
    )

    assert set(axes) == {'scatter', 'middle', 'favorability', 'wins'}
    assert theme[-1] == '0.15'
    plt.close(fig)
