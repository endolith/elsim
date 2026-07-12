"""Geometry, palette, and theme helpers for the two-dimensional examples."""

import importlib
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from scipy.spatial import Voronoi

RESULTS_DIR = Path(__file__).resolve().parent / 'results'
KEY_FRAME_MS = 3000
TRANSITION_TOTAL_MS = 3000

PALETTE_OPTIONS = {
    'palettable.cartocolors.qualitative': [
        'Antique_10', 'Bold_10', 'Pastel_10', 'Prism_10', 'Safe_10', 'Vivid_10',
    ],
    'palettable.colorbrewer.qualitative': [
        'Set3_12', 'Set2_8', 'Set1_9', 'Paired_12', 'Dark2_8', 'Accent_8',
    ],
    'palettable.tableau': [
        'ColorBlind_10', 'GreenOrange_12', 'TableauLight_10', 'TableauMedium_10',
        'Tableau_10', 'Tableau_20',
    ],
    'colorcet': ['glasbey_light', 'glasbey_dark'],
}


def get_palette_colors(name):
    """Load a named palette as a list of Matplotlib-compatible colors."""
    for module_path, names in PALETTE_OPTIONS.items():
        if name in names:
            break
    else:
        raise KeyError(name)

    module = importlib.import_module(module_path)
    palette = getattr(module, name)
    if module_path == 'colorcet':
        return list(palette)
    return list(palette.mpl_colors)


def prepare_palette_and_labels(palette_name, n_cands, dark_background):
    """Load, optionally adjust, and trim a palette and create labels."""
    colors = get_palette_colors(palette_name)
    if not dark_background and palette_name == 'Set1_9' and len(colors) > 5:
        colors.pop(5)
    if n_cands > len(colors):
        raise ValueError(
            f'n_cands={n_cands} exceeds palette "{palette_name}" size '
            f'({len(colors)}). Use fewer candidates or a larger palette.'
        )
    return colors[:n_cands], 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:n_cands]


def _color_to_rgb(color):
    """Normalize a color to an RGB tuple in the [0, 1] range."""
    return mcolors.to_rgb(color)


def remove_grays(colors, min_saturation=0.12):
    """Drop colors with saturation below ``min_saturation``."""
    filtered = []
    for color in colors:
        rgb = np.array(_color_to_rgb(color)).reshape(1, 3)
        if mcolors.rgb_to_hsv(rgb)[0, 1] >= min_saturation:
            filtered.append(color)
    return filtered, len(filtered)


def get_theme(dark_background):
    """Return colors for the dark or light rendering theme."""
    if dark_background:
        return (
            'black',
            'white',
            'white',
            'black',
            'black',
            'white',
            (0.98, 0.98, 0.98),
            '0.15',
        )
    return (
        'white',
        'black',
        'gray',
        'white',
        'white',
        'black',
        (0.12, 0.12, 0.12),
        '0.88',
    )


def transition_step_ms(n_transfer):
    """Return the duration of each non-final transfer frame in milliseconds."""
    return TRANSITION_TOTAL_MS // max(1, n_transfer - 1) if n_transfer > 1 else 0


def candidate_name(candidate_index):
    """Convert a zero-based candidate index to an alphabetical label."""
    return chr(65 + candidate_index)


def ceildiv(a, b):
    """Return the ceiling of integer division for positive integers."""
    return -(-a // b)


def count_wins(matrix):
    """Count strict pairwise wins for every candidate in a comparison matrix."""
    n_cands = matrix.shape[0]
    return [
        sum(matrix[i, j] > matrix[j, i] for j in range(n_cands))
        for i in range(n_cands)
    ]


def voronoi_plot_2d_axes(ax, points, line_color='white', line_alpha=0.45):
    """Draw a Voronoi diagram on an axis without changing its limits."""
    points = np.asarray(points)
    if len(points) < 2:
        return
    if len(points) == 2:
        first, second = points
        delta = second - first
        length = np.linalg.norm(delta)
        if length == 0:
            return
        midpoint = (first + second) / 2
        direction = np.array([-delta[1], delta[0]]) / length
        span = max(np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()), 1.0) * 2
        endpoints = np.array([
            midpoint - span * direction,
            midpoint + span * direction,
        ])
        ax.plot(
            endpoints[:, 0],
            endpoints[:, 1],
            ':',
            color=line_color,
            alpha=line_alpha,
        )
        return

    voronoi = Voronoi(points)
    center = points.mean(axis=0)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    span = max(np.ptp(xlim), np.ptp(ylim))
    finite_segments = []
    infinite_segments = []
    for point_indices, vertices in zip(
        voronoi.ridge_points, voronoi.ridge_vertices
    ):
        vertices = np.asarray(vertices)
        if np.all(vertices >= 0):
            finite_segments.append(voronoi.vertices[vertices])
            continue
        finite_vertex = vertices[vertices >= 0]
        if not len(finite_vertex):
            continue
        tangent = voronoi.points[point_indices[1]] - voronoi.points[point_indices[0]]
        tangent /= np.linalg.norm(tangent)
        normal = np.array([-tangent[1], tangent[0]])
        midpoint = voronoi.points[point_indices].mean(axis=0)
        direction = np.sign(np.dot(midpoint - center, normal)) * normal
        far = voronoi.vertices[finite_vertex[0]] + direction * 2 * span
        infinite_segments.append([voronoi.vertices[finite_vertex[0]], far])

    for segments in (finite_segments, infinite_segments):
        if segments:
            ax.add_collection(
                LineCollection(
                    segments,
                    colors=line_color,
                    lw=1.5,
                    alpha=line_alpha,
                    linestyle=':',
                    zorder=0,
                )
            )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def sort_candidates_bell_curve(candidates):
    """Order left candidates outward-in, center, then right candidates in-out."""
    candidates = np.asarray(candidates)
    distances = np.linalg.norm(candidates, axis=1)
    center_mask = np.isclose(distances, 0)
    left_mask = candidates[:, 0] < 0
    right_mask = ~left_mask & ~center_mask
    center_indices = np.flatnonzero(center_mask)
    left_indices = np.flatnonzero(left_mask)
    right_indices = np.flatnonzero(right_mask)
    left_sorted = left_indices[np.argsort(distances[left_indices])[::-1]]
    right_sorted = right_indices[np.argsort(distances[right_indices])]
    return candidates[
        np.concatenate([left_sorted, center_indices, right_sorted])
    ]
