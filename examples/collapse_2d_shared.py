"""
Shared palette, Voronoi, and theme for 2D collapse finder scripts.

Used by collapse_finder_2d_irv.py and collapse_finder_2d_palette_demo.py.
Blacklist: Pastel2_8, Pastel1_9, BlueRed_12, PurpleGray_12 omitted (always bad).
"""

import importlib
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from scipy.spatial import Voronoi

# (module_path, attr). Uses .mpl_colors except colorcet which uses hex list.
# Pastel2_8, Pastel1_9, BlueRed_12, PurpleGray_12 omitted (always bad).
PALETTE_OPTIONS = {
    'Antique_10': ('palettable.cartocolors.qualitative', 'Antique_10'),
    'Bold_10': ('palettable.cartocolors.qualitative', 'Bold_10'),
    'Pastel_10': ('palettable.cartocolors.qualitative', 'Pastel_10'),
    'Prism_10': ('palettable.cartocolors.qualitative', 'Prism_10'),
    'Safe_10': ('palettable.cartocolors.qualitative', 'Safe_10'),
    'Vivid_10': ('palettable.cartocolors.qualitative', 'Vivid_10'),
    'Set3_12': ('palettable.colorbrewer.qualitative', 'Set3_12'),
    'Set2_8': ('palettable.colorbrewer.qualitative', 'Set2_8'),
    'Set1_9': ('palettable.colorbrewer.qualitative', 'Set1_9'),
    'Paired_12': ('palettable.colorbrewer.qualitative', 'Paired_12'),
    'Dark2_8': ('palettable.colorbrewer.qualitative', 'Dark2_8'),
    'Accent_8': ('palettable.colorbrewer.qualitative', 'Accent_8'),
    'ColorBlind_10': ('palettable.tableau', 'ColorBlind_10'),
    'GreenOrange_12': ('palettable.tableau', 'GreenOrange_12'),
    'TableauLight_10': ('palettable.tableau', 'TableauLight_10'),
    'TableauMedium_10': ('palettable.tableau', 'TableauMedium_10'),
    'Tableau_10': ('palettable.tableau', 'Tableau_10'),
    'Tableau_20': ('palettable.tableau', 'Tableau_20'),
    'TrafficLight_9': ('palettable.tableau', 'TrafficLight_9'),
    'glasbey_light': ('colorcet', 'glasbey_light'),
    'glasbey_dark': ('colorcet', 'glasbey_dark'),
}


def get_palette_colors(name):
    """Load palette as list of colors (mpl tuples or hex)."""
    mod_path, attr = PALETTE_OPTIONS[name]
    mod = importlib.import_module(mod_path)
    pal = getattr(mod, attr)
    if mod_path == 'colorcet':
        return list(pal)
    return list(pal.mpl_colors)


def _color_to_rgb(c):
    """Normalize color to (r, g, b) in [0, 1]."""
    if isinstance(c, str):
        return mcolors.to_rgb(c)
    return tuple(mcolors.to_rgb(c))


def remove_grays(colors, min_saturation=0.12):
    """Drop colors that are effectively gray (low saturation). Returns (filtered_list, n_remaining)."""
    out = []
    for c in colors:
        rgb = np.array(_color_to_rgb(c)).reshape(1, 3)
        hsv = mcolors.rgb_to_hsv(rgb)[0]
        if hsv[1] >= min_saturation:
            out.append(c)
    return out, len(out)


def voronoi_plot_2d_axes(ax, points, line_color='white', line_alpha=0.45):
    """Draw Voronoi diagram of points on ax (no bounds change). Like elsim2k _plotutils."""
    points = np.asarray(points)
    if len(points) < 2:
        return
    if len(points) == 2:
        (x1, y1), (x2, y2) = points[0], points[1]
        ylo, yhi = -100, 100
        xlo = (y2**2 - 2*ylo*y2 - y1**2 + 2*ylo*y1 + x2**2 - x1**2) / (2*x2 - 2*x1)
        xhi = (y2**2 - 2*yhi*y2 - y1**2 + 2*yhi*y1 + x2**2 - x1**2) / (2*x2 - 2*x1)
        ax.plot([xlo, xhi], [ylo, yhi], ':', color=line_color, alpha=line_alpha)
        return
    vor = Voronoi(points)
    center = points.mean(axis=0)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ptp_bound = max(np.ptp(xlim), np.ptp(ylim))

    finite_segments = []
    infinite_segments = []
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            finite_segments.append(vor.vertices[simplex])
        else:
            i = simplex[simplex >= 0][0]
            t = vor.points[pointidx[1]] - vor.points[pointidx[0]]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])
            midpoint = vor.points[pointidx].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far = vor.vertices[i] + direction * 2 * ptp_bound
            infinite_segments.append([vor.vertices[i], far])

    for segs in (finite_segments, infinite_segments):
        if segs:
            lc = LineCollection(segs, colors=line_color, lw=1.5,
                                alpha=line_alpha, linestyle=':', zorder=0)
            ax.add_collection(lc)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def sort_candidates_bell_curve(candidates):
    """
    Reorder candidates so bar graphs form an approximate bell curve:
      left hemisphere (x < 0): farthest → nearest from origin
      the (0, 0) candidate
      right hemisphere (x > 0): nearest → farthest from origin
    "Left" and "right" are determined by the sign of the x-coordinate.
    Candidates exactly at x=0 (other than origin) are treated as right-side.
    """
    dists = np.linalg.norm(candidates, axis=1)
    center_mask = np.all(candidates == 0.0, axis=1)
    left_mask = candidates[:, 0] < 0
    right_mask = ~left_mask & ~center_mask

    center_idx = np.where(center_mask)[0]
    left_idx = np.where(left_mask)[0]
    right_idx = np.where(right_mask)[0]

    left_sorted = left_idx[np.argsort(dists[left_idx])[::-1]]   # farthest first
    right_sorted = right_idx[np.argsort(dists[right_idx])]       # nearest first

    return candidates[np.concatenate([left_sorted, center_idx, right_sorted])]


def get_theme(dark_background):
    """Return (bg, fg, grid, stroke_fg, legend_bg, legend_fg, voronoi_color)."""
    if dark_background:
        return (
            'black', 'white', 'white', 'black', 'black', 'white',
            (0.98, 0.98, 0.98),
        )
    return (
        'white', 'black', 'gray', 'white', 'white', 'black',
        (0.12, 0.12, 0.12),
    )


def setup_scatter_axis_sigma(ax, voters):
    """
    No grid; axis limits and tick labels in units of voter distribution sigma.
    sigma = std(voters). Limits ±1.5*sigma; ticks at ±2σ, ±σ, 0 with σ labels.
    """
    ax.grid(False)
    ax.set_axisbelow(False)
    sigma = float(np.std(voters))
    lim = 1.5 * sigma
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.axis('square')
    tick_pos = [-2 * sigma, -sigma, 0, sigma, 2 * sigma]
    tick_lab = ['−2σ', '−σ', '0', 'σ', '2σ']
    ax.set_xticks(tick_pos)
    ax.set_yticks(tick_pos)
    ax.set_xticklabels(tick_lab)
    ax.set_yticklabels(tick_lab)
