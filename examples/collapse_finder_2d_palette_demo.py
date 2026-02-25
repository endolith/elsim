"""
Demo each colormap as a static first frame (2D scatter + bar chart).

Compares colormaps with similar non-gray color counts: one 7-cand and one 10-cand
election; each scenario reuses the same election so palettes are comparable.
Only palettes with enough non-gray colors are included (>= 7 for 7cand, >= 10 for 10cand).
Saves to Images/palette_demo_<timestamp>/{7,10}cand_{dark,white}/.
"""

import importlib
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np

from elsim.elections import normal_electorate, normed_dist_utilities
from elsim.strategies import honest_rankings

# Same palette options as collapse_finder_2d_irv. We use the maximum-size variant
# per series (Set3_12, Paired_12, Tableau_20, etc.); palettable has no other >8 qualitative.
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
    'glasbey_dark': ('colorcet', 'glasbey_dark'),
}
# Use glasbey_dark on white background (optimized for light bg); glasbey_light on black.
GLASBEY_FOR_WHITE_BG = 'glasbey_dark'


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


def render_first_frame(voters, candidates, ballots, tallies, colors, labels, output_path,
                       dark_background=True):
    """Render IRV start frame (scatter + bar chart)."""
    n_cands = len(candidates)
    n_voters = len(voters)

    if dark_background:
        bg, fg, grid = 'black', 'white', 'white'
        legend_bg, legend_fg = 'black', 'white'
        stroke_fg = 'black'
    else:
        bg, fg, grid = 'white', 'black', 'gray'
        legend_bg, legend_fg = 'white', 'black'
        stroke_fg = 'white'

    fig = plt.figure(figsize=(9, 7.5), facecolor=bg)
    ax_sc = plt.subplot2grid(shape=(4, 3), loc=(0, 0), colspan=2, rowspan=4)
    ax_bar = plt.subplot2grid(shape=(4, 3), loc=(1, 2), rowspan=2)

    for ax in (ax_sc, ax_bar):
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
    ax_sc.grid(True, alpha=0.3, color=grid)
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

    plt.tight_layout()
    plt.savefig(output_path, facecolor=bg, edgecolor='none')
    plt.close(fig)


n_voters = 3000
disp = 0.5

# Compare colormaps with similar non-gray counts: 7 and 10 candidates, same election per n.
N_CAND_SCENARIOS = (7, 10)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_base = Path('Images') / f'palette_demo_{timestamp}'
for n in N_CAND_SCENARIOS:
    for bg in ('dark', 'white'):
        (output_base / f'{n}cand_{bg}').mkdir(parents=True, exist_ok=True)


def prepare_election(n_cands):
    v, c = normal_electorate(n_voters, n_cands, dims=2, disp=disp)
    u = normed_dist_utilities(v, c)
    r = np.asarray(honest_rankings(u))
    b = r[:, 0]
    t = np.bincount(b, minlength=n_cands)
    lbl = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:n_cands]
    return v, c, b, t, lbl


elections = {n: prepare_election(n) for n in N_CAND_SCENARIOS}

def get_colors_for_bg(palette_name, n_cands, dark_background):
    """Get color list for this background. Returns (colors, n_after_grays) or (None, 0)."""
    name = palette_name
    if not dark_background and name == 'glasbey_light':
        name = GLASBEY_FOR_WHITE_BG
    raw = get_palette_colors(name)
    if not dark_background and name == 'Set1_9' and len(raw) > 5:
        c = list(raw)
        c.pop(5)  # Yellow has low visibility on white
        raw = c
    filtered, n_after = remove_grays(raw)
    if len(filtered) < n_cands:
        return None, n_after
    return filtered[:n_cands], n_after

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
        for n_cands in N_CAND_SCENARIOS:
            voters, candidates, ballots, tallies, labels = elections[n_cands]
            for dark in (True, False):
                colors, _ = get_colors_for_bg(palette_name, n_cands, dark)
                if colors is None:
                    continue
                sub = f'{n_cands}cand_{"dark" if dark else "white"}'
                out_path = output_base / sub / f'{palette_name}.png'
                render_first_frame(voters, candidates, ballots, tallies, colors, labels,
                                   out_path, dark_background=dark)
                print(f'Saved {out_path}')
    except Exception as e:
        print(f'Failed {palette_name}: {e}')

print(f'Done. Images in {output_base.resolve()}')
