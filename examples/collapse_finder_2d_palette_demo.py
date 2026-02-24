"""
Demo each colormap as a static first frame (2D scatter + bar chart).

Generates one random election and renders the initial IRV state for each
palette in PALETTE_OPTIONS. Saves images to Images/palette_demo_<timestamp>/.
"""

import importlib
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np

from elsim.elections import normal_electorate, normed_dist_utilities
from elsim.strategies import honest_rankings

# Same palette options as collapse_finder_2d_irv
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
    'Pastel2_8': ('palettable.colorbrewer.qualitative', 'Pastel2_8'),
    'Pastel1_9': ('palettable.colorbrewer.qualitative', 'Pastel1_9'),
    'Paired_12': ('palettable.colorbrewer.qualitative', 'Paired_12'),
    'Dark2_8': ('palettable.colorbrewer.qualitative', 'Dark2_8'),
    'Accent_8': ('palettable.colorbrewer.qualitative', 'Accent_8'),
    'BlueRed_12': ('palettable.tableau', 'BlueRed_12'),
    'ColorBlind_10': ('palettable.tableau', 'ColorBlind_10'),
    'GreenOrange_12': ('palettable.tableau', 'GreenOrange_12'),
    'PurpleGray_12': ('palettable.tableau', 'PurpleGray_12'),
    'TableauLight_10': ('palettable.tableau', 'TableauLight_10'),
    'TableauMedium_10': ('palettable.tableau', 'TableauMedium_10'),
    'Tableau_10': ('palettable.tableau', 'Tableau_10'),
    'Tableau_20': ('palettable.tableau', 'Tableau_20'),
    'TrafficLight_9': ('palettable.tableau', 'TrafficLight_9'),
    'glasbey_light': ('colorcet', 'glasbey_light'),
}


def get_palette_colors(name):
    """Load palette as list of colors (mpl tuples or hex)."""
    mod_path, attr = PALETTE_OPTIONS[name]
    mod = importlib.import_module(mod_path)
    pal = getattr(mod, attr)
    if mod_path == 'colorcet':
        return list(pal)
    return list(pal.mpl_colors)


def render_first_frame(voters, candidates, ballots, tallies, colors, labels, output_path):
    """Render IRV start frame (scatter + bar chart) with dark background."""
    n_cands = len(candidates)
    n_voters = len(voters)

    fig = plt.figure(figsize=(9, 7.5), facecolor='black')
    ax_sc = plt.subplot2grid(shape=(4, 3), loc=(0, 0), colspan=2, rowspan=4)
    ax_bar = plt.subplot2grid(shape=(4, 3), loc=(1, 2), rowspan=2)

    for ax in (ax_sc, ax_bar):
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('white')

    voters_kwargs = {'marker': '.', 'alpha': 0.25, 's': 12}
    cands_kwargs = {'marker': 'o', 's': 30, 'edgecolors': 'white'}
    path_effects = [PathEffects.withStroke(linewidth=3, foreground='black')]

    ax_sc.scatter([], [], color='w', **voters_kwargs, label='Voters')
    ax_sc.scatter([], [], color='w', **cands_kwargs, label='Candidates')
    ax_sc.legend(loc='lower right', numpoints=1, fontsize='small', labelcolor='white',
                 facecolor='black', edgecolor='white')
    ax_sc.grid(True, alpha=0.3, color='white')
    ax_sc.set_axisbelow(True)
    ax_sc.axis('square')
    ax_sc.axis([-3, 3, -3, 3])

    for cand in range(n_cands):
        cand_voters = voters[ballots == cand]
        if len(cand_voters):
            ax_sc.scatter(cand_voters[:, 0], cand_voters[:, 1], color=colors[cand], **voters_kwargs)
    ax_sc.scatter(candidates[:, 0], candidates[:, 1], color=colors, **cands_kwargs)
    for cand, pos in enumerate(candidates):
        ax_sc.annotate(labels[cand], xy=pos, xytext=(0, -15),
                       textcoords='offset points', path_effects=path_effects, color='white')

    bars = ax_bar.bar(range(n_cands), tallies / n_voters * 100, tick_label=list(labels), color=colors)
    for rect in bars:
        height = rect.get_height()
        if height > 0:
            ax_bar.annotate(f'{height:.0f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', color='white')
    ax_bar.set_ylim(0, 100)
    ax_bar.set_ylabel('Votes [%]')
    ax_bar.grid(True, alpha=0.25, axis='y', color='white')
    ax_bar.set_axisbelow(True)
    ax_bar.text(0.5, 1.04, 'IRV start', transform=ax_bar.transAxes, ha='center', va='center', color='white')

    plt.tight_layout()
    plt.savefig(output_path, facecolor='black', edgecolor='none')
    plt.close(fig)


n_voters = 3000
n_cands = 8
disp = 0.5

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = Path('Images') / f'palette_demo_{timestamp}'
output_dir.mkdir(parents=True, exist_ok=True)

voters, candidates = normal_electorate(n_voters, n_cands, dims=2, disp=disp)
utilities = normed_dist_utilities(voters, candidates)
rankings = np.asarray(honest_rankings(utilities))
ballots = rankings[:, 0]
tallies = np.bincount(ballots, minlength=n_cands)
labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:n_cands]

for palette_name in PALETTE_OPTIONS:
    try:
        colors = get_palette_colors(palette_name)[:n_cands]
        out_path = output_dir / f'{palette_name}.png'
        render_first_frame(voters, candidates, ballots, tallies, colors, labels, out_path)
        print(f'Saved {out_path}')
    except Exception as e:
        print(f'Failed {palette_name}: {e}')

print(f'Done. Images in {output_dir.resolve()}')
