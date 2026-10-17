"""
Shared configuration, palette, theme, plotting helpers and election sampling
for the 2D collapse finder scripts.

Used by collapse_finder_2d_irv.py, collapse_finder_2d_tvr.py,
collapse_finder_2d_both.py and collapse_finder_2d_palette_demo.py.

Palette blacklist: Pastel2_8, Pastel1_9, BlueRed_12, PurpleGray_12 and
TrafficLight_9 are omitted (always bad).
"""

import argparse
import importlib
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from scipy.spatial import Voronoi

from elsim.elections import normal_electorate, normed_dist_utilities
from elsim.strategies import honest_rankings

# Default output root for collapse finder scripts (examples/results/).
RESULTS_DIR = Path(__file__).resolve().parent / 'results'

CANDIDATE_LABELS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


# ── Configuration ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class AnimationConfig:
    """Everything the IRV/TVR/both scripts need to search and render."""
    n_voters: int = 5000
    n_cands: int = 9
    max_trials: int = 100_000
    frames_per_transfer: int = 60
    disp: float = 0.5
    dark_background: bool = True
    palette_name: str = 'Bold_10'
    seed: int | None = None
    # GIF timing: key frames (start, eliminate, end of transition, winner)
    # are held for key_frame_ms; a whole transition lasts transition_total_ms.
    key_frame_ms: int = 3000
    transition_total_ms: int = 3000

    def transition_step_ms(self):
        """Per-frame ms so a full transition lasts ``transition_total_ms``."""
        n = self.frames_per_transfer
        return self.transition_total_ms // max(1, n - 1) if n > 1 else 0

    def labels(self):
        return CANDIDATE_LABELS[:self.n_cands]


DEFAULT_CONFIG = AnimationConfig()


def parse_args(description, *, allow_positions=False):
    """
    Command-line interface shared by the collapse finder scripts.

    Returns
    -------
    config : AnimationConfig
    positions : Path or None
        Only when ``allow_positions`` is True: a ``positions.npz`` from an
        earlier run to reuse instead of searching for a new election.
    """
    p = argparse.ArgumentParser(description=description)
    d = DEFAULT_CONFIG
    p.add_argument('--n-voters', type=int, default=d.n_voters)
    p.add_argument('--n-cands', type=int, default=d.n_cands)
    p.add_argument('--max-trials', type=int, default=d.max_trials)
    p.add_argument('--frames-per-transfer', type=int,
                   default=d.frames_per_transfer)
    p.add_argument('--disp', type=float, default=d.disp,
                   help='candidate dispersion relative to voters')
    p.add_argument('--palette', default=d.palette_name, choices=PALETTE_NAMES)
    p.add_argument('--light', action='store_true',
                   help='render on a white background')
    p.add_argument('--seed', type=int, default=None,
                   help='RNG seed for reproducible search and animation')
    if allow_positions:
        p.add_argument('--positions', type=Path, default=None,
                       help='positions.npz from a previous run to reuse')
    a = p.parse_args()
    config = replace(
        d, n_voters=a.n_voters, n_cands=a.n_cands, max_trials=a.max_trials,
        frames_per_transfer=a.frames_per_transfer, disp=a.disp,
        palette_name=a.palette, dark_background=not a.light, seed=a.seed)
    if allow_positions:
        return config, a.positions
    return config


# ── Elections ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Election:
    """A spatial election: positions plus the honest ranked ballots."""
    voters: np.ndarray       # (n_voters, 2)
    candidates: np.ndarray   # (n_cands, 2)
    rankings: np.ndarray     # (n_voters, n_cands)

    @classmethod
    def from_positions(cls, voters, candidates):
        utilities = normed_dist_utilities(voters, candidates)
        return cls(voters, candidates, np.asarray(honest_rankings(utilities)))

    @property
    def n_voters(self):
        return len(self.voters)

    @property
    def n_cands(self):
        return len(self.candidates)

    def distances_to_origin(self):
        return np.linalg.norm(self.candidates, axis=1)

    def center_candidate(self):
        """Index of the candidate nearest the origin (the Condorcet winner)."""
        return int(np.argmin(self.distances_to_origin()))

    def approval_pct(self):
        """Mean normalised utility per candidate, 0–100."""
        utilities = normed_dist_utilities(self.voters, self.candidates)
        return utilities.mean(axis=0) * 100

    def save_positions(self, path):
        np.savez(path, voters=self.voters, candidates=self.candidates)


def load_election(positions_path):
    """Rebuild an `Election` from a saved ``positions.npz``."""
    data = np.load(positions_path)
    return Election.from_positions(data['voters'], data['candidates'])


def sample_election(config):
    """One random 2D election with a candidate exactly at the origin."""
    voters, candidates = normal_electorate(
        config.n_voters, config.n_cands, dims=2, disp=config.disp)
    candidates[0] = 0.0
    candidates = sort_candidates_bell_curve(candidates)
    return Election.from_positions(voters, candidates)


def find_election(config, analyse):
    """
    Sample random elections until ``analyse(election)`` returns a result.

    ``analyse`` returns ``None`` to reject an election and anything else
    (typically a trace) to accept it.

    Returns
    -------
    (trial, election, result) or None if ``max_trials`` is exhausted.
    """
    if config.seed is not None:
        np.random.seed(config.seed)
    for trial in range(1, config.max_trials + 1):
        election = sample_election(config)
        result = analyse(election)
        if result is not None:
            return trial, election, result
    return None


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

    left_sorted = left_idx[np.argsort(dists[left_idx])[::-1]]  # farthest 1st
    right_sorted = right_idx[np.argsort(dists[right_idx])]     # nearest 1st

    return candidates[np.concatenate([left_sorted, center_idx, right_sorted])]


def candidate_name(candidate_index):
    """Convert candidate index to name (A, B, C, etc.)."""
    return CANDIDATE_LABELS[candidate_index]


def ceildiv(a, b):
    """Ceiling division for positive integers."""
    return -(-a // b)


# ── Palettes ──────────────────────────────────────────────────────────

# module_path -> list of palette attr names.
# Uses .mpl_colors except colorcet (hex list).
PALETTE_OPTIONS = {
    'palettable.cartocolors.qualitative': [
        'Antique_10', 'Bold_10', 'Pastel_10',
        'Prism_10', 'Safe_10', 'Vivid_10',
    ],
    'palettable.colorbrewer.qualitative': [
        'Set3_12', 'Set2_8', 'Set1_9',
        'Paired_12', 'Dark2_8', 'Accent_8',
    ],
    'palettable.tableau': [
        'ColorBlind_10', 'GreenOrange_12',
        'TableauLight_10', 'TableauMedium_10',
        'Tableau_10', 'Tableau_20',
    ],
    'colorcet': ['glasbey_light', 'glasbey_dark'],
}

PALETTE_NAMES = [n for names in PALETTE_OPTIONS.values() for n in names]

GRAY = (0.5, 0.5, 0.5)


def get_palette_colors(name):
    """Load palette as list of colors (mpl tuples or hex)."""
    for mod_path, names in PALETTE_OPTIONS.items():
        if name in names:
            break
    else:
        raise KeyError(name)
    mod = importlib.import_module(mod_path)
    pal = getattr(mod, name)
    if mod_path == 'colorcet':
        return list(pal)
    return list(pal.mpl_colors)


def prepare_palette(palette_name, n_cands, dark_background):
    """
    Load palette, apply Set1_9 yellow fix for light bg, trim to n_cands.
    Raises ValueError if palette too small.
    """
    colors = get_palette_colors(palette_name)
    if not dark_background and palette_name == 'Set1_9' and len(colors) > 5:
        colors.pop(5)  # yellow is invisible on white
    if n_cands > len(colors):
        raise ValueError(
            f'n_cands={n_cands} exceeds palette "{palette_name}" size '
            f'({len(colors)}). Use fewer candidates or a larger palette.')
    return colors[:n_cands]


def prepare_palette_and_labels(palette_name, n_cands, dark_background):
    """Backwards-compatible helper returning ``(colors, labels)``."""
    return (prepare_palette(palette_name, n_cands, dark_background),
            CANDIDATE_LABELS[:n_cands])


def _color_to_rgb(c):
    """Normalize color to (r, g, b) in [0, 1]."""
    return tuple(mcolors.to_rgb(c))


def remove_grays(colors, min_saturation=0.12):
    """Drop colors that are effectively gray (low saturation).

    Returns (filtered_list, n_remaining).
    """
    out = []
    for c in colors:
        rgb = np.array(_color_to_rgb(c)).reshape(1, 3)
        hsv = mcolors.rgb_to_hsv(rgb)[0]
        if hsv[1] >= min_saturation:
            out.append(c)
    return out, len(out)


# ── Theme ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Theme:
    bg: str
    fg: str
    grid: str
    stroke_fg: str
    legend_bg: str
    legend_fg: str
    voronoi_color: tuple
    dead_zone_color: str  # band for TVR rank dead zone

    def style_axes(self, *axes):
        for ax in axes:
            ax.set_facecolor(self.bg)
            ax.tick_params(colors=self.fg)
            ax.xaxis.label.set_color(self.fg)
            ax.yaxis.label.set_color(self.fg)
            for spine in ax.spines.values():
                spine.set_color(self.fg)


DARK_THEME = Theme('black', 'white', 'white', 'black', 'black', 'white',
                   (0.98, 0.98, 0.98), '0.15')
# A dark dead-zone band looks wrong on white, so use light gray there.
LIGHT_THEME = Theme('white', 'black', 'gray', 'white', 'white', 'black',
                    (0.12, 0.12, 0.12), '0.88')


def get_theme(dark_background):
    return DARK_THEME if dark_background else LIGHT_THEME


# ── Plot helpers ──────────────────────────────────────────────────────

def annotate_bars(ax, bars, fmt='{:.0f}', color='black'):
    """Write each bar's height above it (skipping empty bars)."""
    for rect in bars:
        height = rect.get_height()
        if height > 0:
            ax.annotate(fmt.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', color=color)


def set_panel_title(ax, title, color):
    """Small centred title just above a subplot."""
    ax.text(0.5, 1.04, title, transform=ax.transAxes,
            ha='center', va='center', color=color)


def plot_wins(ax, wins, colors, labels, edgecolor='black', gap=0.15):
    """Plot head-to-head wins as stacked square blocks per candidate.
    Blocks are square, symmetrically spaced; integer y-ticks."""
    n_cands = len(wins)
    block = 1.0 - 2 * gap
    max_w = max(wins) if wins else 0
    for n in range(n_cands):
        for i in range(int(wins[n])):
            ax.bar(n, block, bottom=i + gap, width=block,
                   color=colors[n], edgecolor=edgecolor, linewidth=1)
    ax.set_xticks(range(n_cands))
    ax.set_xticklabels(list(labels))
    ax.set_xlim(-0.5, n_cands - 0.5)
    ax.set_ylim(0, max_w if max_w > 0 else 1)
    ax.set_aspect('equal')
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_ylabel('')


def plot_approval_bar(ax, approval_pct, labels, colors, theme):
    """Bar chart for approval rating 0–100%."""
    bars = ax.bar(range(len(labels)), approval_pct,
                  tick_label=list(labels), color=colors)
    annotate_bars(ax, bars, color=theme.fg)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Approval [%]')
    ax.grid(True, alpha=0.25, axis='y', color=theme.grid)
    ax.set_axisbelow(True)
    set_panel_title(ax, 'Approval rating', theme.fg)


def plot_wins_with_title(ax, wins, colors, labels, theme, gap=0.1):
    """Plot head-to-head wins and set subplot title."""
    plot_wins(ax, wins, colors, labels, edgecolor=theme.fg, gap=gap)
    set_panel_title(ax, 'Head-to-head wins', theme.fg)


def voronoi_plot_2d_axes(ax, points, line_color='white', line_alpha=0.45):
    """Draw Voronoi diagram of points on ax without changing its bounds."""
    points = np.asarray(points)
    if len(points) < 2:
        return
    if len(points) == 2:
        (x1, y1), (x2, y2) = points[0], points[1]
        ylo, yhi = -100, 100
        xlo = ((y2**2 - 2*ylo*y2 - y1**2 + 2*ylo*y1
                + x2**2 - x1**2) / (2*x2 - 2*x1))
        xhi = ((y2**2 - 2*yhi*y2 - y1**2 + 2*yhi*y1
                + x2**2 - x1**2) / (2*x2 - 2*x1))
        ax.plot([xlo, xhi], [ylo, yhi], ':',
                color=line_color, alpha=line_alpha)
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
