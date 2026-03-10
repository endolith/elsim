"""
Demo each colormap as a static first frame matching the IRV animation layout.

Uses a fixed election (positions.npz) so all palettes are comparable. Layout:
scatter + votes bar + approval rating + head-to-head wins (same as IRV start frame).
Saves to Images/palette_demo_<timestamp>/{9cand_dark,9cand_white}/<palette>.png.
"""

from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np

from elsim.elections import normed_dist_utilities
from elsim.methods import ranked_election_to_matrix
from elsim.strategies import honest_rankings

from collapse_2d_shared import (
    PALETTE_OPTIONS,
    get_palette_colors,
    get_theme,
    remove_grays,
    setup_scatter_axis_sigma,
    voronoi_plot_2d_axes,
)
from collapse_utils import count_wins
from collapse_finder_2d_irv import plot_wins


INPUT_POSITIONS = Path('Images/collapse_2d_both_20260308_141324_nc9_nv5000 great/positions.npz')


def render_first_frame(voters, candidates, ballots, tallies, approval_pct, wins,
                       colors, labels, output_path, dark_background=True):
    """Render IRV start frame: same (6,3) layout as collapse_finder_2d_irv (scatter, votes, approval, head-to-head)."""
    n_cands = len(candidates)
    n_voters = len(voters)

    bg, fg, grid, stroke_fg, legend_bg, legend_fg, voronoi_color, _ = get_theme(dark_background)

    fig = plt.figure(figsize=(9, 7.5), facecolor=bg)
    ax_sc = plt.subplot2grid(shape=(6, 3), loc=(0, 0), colspan=2, rowspan=6)
    ax_bar = plt.subplot2grid(shape=(6, 3), loc=(0, 2), rowspan=2)
    ax_score = plt.subplot2grid(shape=(6, 3), loc=(2, 2), rowspan=2)
    ax_wins = plt.subplot2grid(shape=(6, 3), loc=(4, 2), rowspan=2)

    for ax in (ax_sc, ax_bar, ax_score, ax_wins):
        ax.set_facecolor(bg)
        ax.tick_params(colors=fg)
        ax.xaxis.label.set_color(fg)
        ax.yaxis.label.set_color(fg)
        for spine in ax.spines.values():
            spine.set_color(fg)

    voters_kwargs = {'marker': '.', 'alpha': 0.25, 's': 12}
    cands_kwargs = {'marker': 'o', 's': 30, 'edgecolors': fg}
    path_effects = [PathEffects.withStroke(linewidth=3, foreground=stroke_fg)]

    ax_sc.scatter([], [], color=fg, **voters_kwargs, label='Voters')
    ax_sc.scatter([], [], color=fg, **cands_kwargs, label='Candidates')
    ax_sc.legend(loc='lower right', numpoints=1, fontsize='small', labelcolor=legend_fg,
                 facecolor=legend_bg, edgecolor=legend_fg)
    setup_scatter_axis_sigma(ax_sc, voters)

    voronoi_plot_2d_axes(ax_sc, candidates, line_color=voronoi_color, line_alpha=0.45)

    for cand in range(n_cands):
        cand_voters = voters[ballots == cand]
        if len(cand_voters):
            ax_sc.scatter(cand_voters[:, 0], cand_voters[:, 1], color=colors[cand], **voters_kwargs)
    ax_sc.scatter(candidates[:, 0], candidates[:, 1], color=colors, **cands_kwargs)
    for cand, pos in enumerate(candidates):
        ax_sc.annotate(labels[cand], xy=pos, xytext=(0, -15),
                       textcoords='offset points', path_effects=path_effects, color=fg)

    bars = ax_bar.bar(range(n_cands), tallies / n_voters * 100, tick_label=list(labels), color=colors)
    for rect in bars:
        height = rect.get_height()
        if height > 0:
            ax_bar.annotate(f'{height:.0f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', color=fg)
    ax_bar.set_ylim(0, 100)
    ax_bar.set_ylabel('Votes [%]')
    ax_bar.grid(True, alpha=0.25, axis='y', color=grid)
    ax_bar.set_axisbelow(True)
    ax_bar.text(0.5, 1.04, 'IRV start', transform=ax_bar.transAxes, ha='center', va='center', color=fg)

    score_bars = ax_score.bar(range(n_cands), approval_pct, tick_label=list(labels), color=colors)
    for rect in score_bars:
        height = rect.get_height()
        if height > 0:
            ax_score.annotate(f'{height:.0f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                              xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', color=fg)
    ax_score.set_ylim(0, 100)
    ax_score.set_ylabel('Approval [%]')
    ax_score.grid(True, alpha=0.25, axis='y', color=grid)
    ax_score.set_axisbelow(True)
    ax_score.text(0.5, 1.04, 'Approval rating', transform=ax_score.transAxes, ha='center', va='center', color=fg)

    plot_wins(ax_wins, wins, colors, labels, edgecolor=fg, gap=0.1)
    ax_wins.text(0.5, 1.04, 'Head-to-head wins', transform=ax_wins.transAxes, ha='center', va='center', color=fg)

    plt.tight_layout()
    plt.savefig(output_path, facecolor=bg, edgecolor='none')
    plt.close(fig)


def get_colors_for_bg(palette_name, n_cands, dark_background):
    """Get color list for this background. Returns (colors, n_after_grays) or (None, 0)."""
    raw = get_palette_colors(palette_name)
    if not dark_background and palette_name == 'Set1_9' and len(raw) > 5:
        c = list(raw)
        c.pop(5)
        raw = c
    filtered, n_after = remove_grays(raw)
    if len(filtered) < n_cands:
        return None, n_after
    return filtered[:n_cands], n_after


if __name__ == '__main__':
    data = np.load(INPUT_POSITIONS)
    voters = data['voters']
    candidates = data['candidates']
    n_cands = len(candidates)
    n_voters = len(voters)

    utilities = normed_dist_utilities(voters, candidates)
    rankings = np.asarray(honest_rankings(utilities))
    ballots = rankings[:, 0]
    tallies = np.bincount(ballots, minlength=n_cands)
    approval_pct = utilities.mean(axis=0) * 100
    wins = count_wins(ranked_election_to_matrix(rankings))
    labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:n_cands]

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_base = Path('Images') / f'palette_demo_{timestamp}'
    for sub in ('9cand_dark', '9cand_white'):
        (output_base / sub).mkdir(parents=True, exist_ok=True)

    print(f'Loaded election from {INPUT_POSITIONS} ({n_cands} candidates, {n_voters} voters).')
    print('Palette sizes (original -> after removing grays):')
    for pname in PALETTE_OPTIONS:
        try:
            raw = get_palette_colors(pname)
            filtered, n = remove_grays(raw)
            print(f'  {pname}: {len(raw)} -> {n} non-gray')
        except Exception as e:
            print(f'  {pname}: failed ({e})')
    print()

    for palette_name in PALETTE_OPTIONS:
        try:
            for dark in (True, False):
                if not dark and palette_name == 'glasbey_light':
                    continue
                if dark and palette_name == 'glasbey_dark':
                    continue
                if dark and palette_name == 'Safe_10':
                    continue
                if not dark and palette_name in ('Pastel_10', 'Set3_12'):
                    continue
                colors, _ = get_colors_for_bg(palette_name, n_cands, dark)
                if colors is None:
                    continue
                sub = '9cand_dark' if dark else '9cand_white'
                out_path = output_base / sub / f'{palette_name}.png'
                render_first_frame(
                    voters, candidates, ballots, tallies, approval_pct, wins,
                    colors, labels, out_path, dark_background=dark,
                )
                print(f'Saved {out_path}')
        except Exception as e:
            print(f'Failed {palette_name}: {e}')

    print(f'Done. Images in {output_base.resolve()}')
