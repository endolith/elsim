"""Shared geometry, palette, and rendering helpers for 2D collapse examples."""

import importlib
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patheffects as PathEffects
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

PALETTE_NAMES = [
    name for names in PALETTE_OPTIONS.values() for name in names
]


def transition_step_ms(n_transfer):
    """Return the duration of each non-final transfer frame in milliseconds."""
    return TRANSITION_TOTAL_MS // max(1, n_transfer - 1) if n_transfer > 1 else 0


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


def plot_wins(ax, wins, colors, labels, edgecolor='black', gap=0.15):
    """Plot strict head-to-head wins as stacked square blocks."""
    n_cands = len(wins)
    block = 1.0 - 2 * gap
    max_wins = max(wins) if wins else 0
    for candidate in range(n_cands):
        for index in range(int(wins[candidate])):
            ax.bar(
                candidate,
                block,
                bottom=index + gap,
                width=block,
                color=colors[candidate],
                edgecolor=edgecolor,
                linewidth=1,
            )
    ax.set_xticks(range(n_cands))
    ax.set_xticklabels(list(labels))
    ax.set_xlim(-0.5, n_cands - 0.5)
    ax.set_ylim(0, max_wins if max_wins > 0 else 1)
    ax.set_aspect('equal')
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_ylabel('')


def plot_wins_with_title(ax, wins, colors, labels, fg, gap=0.1):
    """Plot head-to-head wins and add the standard panel title."""
    plot_wins(ax, wins, colors, labels, edgecolor=fg, gap=gap)
    ax.text(
        0.5,
        1.04,
        'Head-to-head wins',
        transform=ax.transAxes,
        ha='center',
        va='center',
        color=fg,
    )


def plot_favorability_bar(ax, favorability_pct, labels, colors, fg, grid):
    """Plot mean normalized utility as an average favorability percentage."""
    bars = ax.bar(
        range(len(labels)),
        favorability_pct,
        tick_label=list(labels),
        color=colors,
    )
    for rect in bars:
        height = rect.get_height()
        if height > 0:
            ax.annotate(
                f'{height:.0f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords='offset points',
                ha='center',
                va='bottom',
                color=fg,
            )
    ax.set_ylim(0, 100)
    ax.set_ylabel('Mean utility [%]')
    ax.grid(True, alpha=0.25, axis='y', color=grid)
    ax.set_axisbelow(True)
    ax.text(
        0.5,
        1.04,
        'Average favorability',
        transform=ax.transAxes,
        ha='center',
        va='center',
        color=fg,
    )


def prepare_palette_and_labels(palette_name, n_cands, dark_background):
    """Load, optionally adjust, and trim a palette and create candidate labels."""
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
        endpoints = np.array([midpoint - span * direction, midpoint + span * direction])
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
    center_mask = np.all(candidates == 0.0, axis=1)
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


def setup_scatter_axis_sigma(ax, voters):
    """Set square limits and visible ticks in units of voter-distribution σ."""
    ax.grid(False)
    ax.set_axisbelow(False)
    sigma = float(np.std(voters))
    limit = 1.5 * sigma
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.axis('square')
    tick_positions = [-sigma, 0, sigma]
    tick_labels = ['−σ', '0', 'σ']
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)


def create_frame_scaffold(
    voters,
    candidates,
    ballots,
    favorability_pct,
    wins,
    colors,
    labels,
    eliminated=None,
    dark_background=True,
):
    """Create the common four-panel figure used by collapse animations.

    The returned ``axes['middle']`` is intentionally left empty for the caller
    to populate with method-specific vote or score data.
    """
    eliminated = set() if eliminated is None else set(eliminated)
    n_cands = len(candidates)
    active = [candidate for candidate in range(n_cands) if candidate not in eliminated]
    active_colors = [
        colors[candidate] if candidate not in eliminated else (0.5, 0.5, 0.5)
        for candidate in range(n_cands)
    ]
    (
        bg,
        fg,
        grid,
        stroke_fg,
        legend_bg,
        legend_fg,
        voronoi_color,
        dead_zone_color,
    ) = get_theme(dark_background)

    fig = plt.figure(figsize=(9, 7.5), facecolor=bg)
    axes = {
        'scatter': plt.subplot2grid((6, 3), (0, 0), colspan=2, rowspan=6),
        'middle': plt.subplot2grid((6, 3), (0, 2), rowspan=2),
        'favorability': plt.subplot2grid((6, 3), (2, 2), rowspan=2),
        'wins': plt.subplot2grid((6, 3), (4, 2), rowspan=2),
    }
    for axis in axes.values():
        axis.set_facecolor(bg)
        axis.tick_params(colors=fg)
        axis.xaxis.label.set_color(fg)
        axis.yaxis.label.set_color(fg)
        for spine in axis.spines.values():
            spine.set_color(fg)

    voters_kwargs = {'marker': '.', 'alpha': 0.25, 's': 12}
    candidates_kwargs = {'marker': 'o', 's': 30, 'edgecolors': fg}
    axes['scatter'].scatter(
        [], [], color=fg, **voters_kwargs, label='Voters'
    )
    axes['scatter'].scatter(
        [], [], color=fg, **candidates_kwargs, label='Candidates'
    )
    axes['scatter'].legend(
        loc='lower right',
        numpoints=1,
        fontsize='small',
        labelcolor=legend_fg,
        facecolor=legend_bg,
        edgecolor=legend_fg,
    )
    setup_scatter_axis_sigma(axes['scatter'], voters)
    voronoi_plot_2d_axes(
        axes['scatter'],
        candidates[active],
        line_color=voronoi_color,
        line_alpha=0.45,
    )

    path_effects = [
        PathEffects.withStroke(linewidth=3, foreground=stroke_fg)
    ]
    for candidate in range(n_cands):
        candidate_voters = voters[ballots == candidate]
        if len(candidate_voters):
            axes['scatter'].scatter(
                candidate_voters[:, 0],
                candidate_voters[:, 1],
                color=active_colors[candidate],
                **voters_kwargs,
            )
    if active:
        axes['scatter'].scatter(
            candidates[active, 0],
            candidates[active, 1],
            color=[active_colors[candidate] for candidate in active],
            **candidates_kwargs,
        )
        for candidate in active:
            axes['scatter'].annotate(
                labels[candidate],
                xy=candidates[candidate],
                xytext=(0, -15),
                textcoords='offset points',
                path_effects=path_effects,
                color=fg,
            )

    plot_favorability_bar(
        axes['favorability'],
        favorability_pct,
        labels,
        active_colors,
        fg,
        grid,
    )
    plot_wins_with_title(axes['wins'], wins, active_colors, labels, fg)
    return fig, axes, active_colors, (bg, fg, grid, dead_zone_color)
