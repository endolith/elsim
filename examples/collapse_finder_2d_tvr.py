"""
Find a 2D center-outward IRV collapse and animate the same election under TVR
(Total Vote Runoff = Baldwin's method).

TVR eliminates the candidate with the lowest Borda score each round, where
Borda scores are re-tallied among remaining candidates only.  Unlike IRV, TVR
satisfies the Condorcet criterion: the center candidate (Condorcet winner) is
never eliminated and always wins.  The same election that shows IRV's center
collapse shows TVR's center convergence.

Set INPUT_POSITIONS to the positions.npz path saved by collapse_finder_2d_irv.py
to reuse that election, or leave None to search for a fresh one.
"""

from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from elsim.methods import ranked_election_to_matrix
from elsim.strategies import honest_rankings
from elsim.elections import normal_electorate, normed_dist_utilities

from collapse_2d_shared import (
    PALETTE_OPTIONS,
    get_palette_colors,
    get_theme,
    setup_scatter_axis_sigma,
    sort_candidates_bell_curve,
    voronoi_plot_2d_axes,
)
from collapse_utils import count_wins


# ── Configuration ────────────────────────────────────────────────────────────

# Path to positions.npz from IRV run, or None to search
INPUT_POSITIONS = r".\Images\collapse_2d_irv_20260307_005817_nc9_nv5000\positions.npz"

palette_name = 'Bold_10'

n_voters = 5000
n_cands = 9
max_trials = 100_000
frames_per_transfer = 60
disp = 0.5
dark_background = True


# ── Helpers ───────────────────────────────────────────────────────────────────

def candidate_name(candidate_index):
    """Convert candidate index to name (A, B, C, etc.)."""
    return chr(65 + candidate_index)


def ceildiv(a, b):
    """Ceiling division for positive integers."""
    return -(-a // b)


def compute_borda_scores(rankings, active):
    """
    Borda scores for active candidates as if non-active were never on the ballot.

    Rankings: n_voters × n_cands array of candidate indices (0-based).
    Returns ndarray of length n_cands; eliminated candidates have score 0.
    Scoring: n_remaining-1 points for 1st place, 0 for last place.
    """
    active_set = set(active)
    n_remaining = len(active)
    scores = np.zeros(rankings.shape[1], dtype=float)
    for ballot in rankings:
        pos = 0
        for cand_id in ballot:
            if cand_id in active_set:
                scores[cand_id] += (n_remaining - 1 - pos)
                pos += 1
    return scores


def simulate_tvr_rounds(rankings, candidates):
    """
    Simulate TVR (Baldwin's method) with round-by-round trace data.

    Each round eliminates the candidate with the lowest Borda score among
    remaining candidates (re-tallied as if eliminated candidates were never
    on the ballot).  Returns None on any tie so only clean runs are kept.

    Per-round trace includes enough data to animate the ballot updates voter
    by voter: for each voter, which active candidates move up in rank when
    the loser is removed (i.e. which candidates are ranked below the loser
    in that voter's ballot).
    """
    n_voters, n_cands = rankings.shape
    active = list(range(n_cands))
    rounds = []

    while len(active) > 1:
        n_remaining = len(active)
        borda_before = compute_borda_scores(rankings, active)

        min_score = min(borda_before[c] for c in active)
        low_scorers = [c for c in active if borda_before[c] == min_score]
        if len(low_scorers) != 1:
            return None

        loser = low_scorers[0]

        # For each voter: the active candidates ranked below the loser.
        # When loser is removed, each of these gains +1 Borda point.
        active_set = set(active)
        promoted_per_voter = []
        for ballot in rankings:
            promoted = []
            found_loser = False
            for cand_id in ballot:
                if cand_id not in active_set:
                    continue
                if cand_id == loser:
                    found_loser = True
                    continue
                if found_loser:
                    promoted.append(cand_id)
            promoted_per_voter.append(promoted)

        active.remove(loser)
        borda_after = compute_borda_scores(rankings, active)

        rounds.append({
            'loser': loser,
            'borda_before': borda_before,
            'borda_after': borda_after,
            'promoted_per_voter': promoted_per_voter,
        })

    winner = active[0]
    # Verify winner is nearest to origin (the Condorcet/center candidate).
    dists = np.linalg.norm(candidates, axis=1)
    center = int(np.argmin(dists))
    if winner != center:
        return None  # TVR didn't converge to center candidate; skip this election

    return {
        'rounds': rounds,
        'winner': winner,
    }


def find_center_convergent_election(n_voters, n_cands, max_trials, disp=1.0):
    """
    Sample random 2D elections until TVR converges to the center candidate.

    Returns (trial, voters, candidates, rankings, trace).
    """
    for trial in range(1, max_trials + 1):
        voters, candidates = normal_electorate(n_voters, n_cands, dims=2, disp=disp)
        candidates[0] = 0.0
        candidates = sort_candidates_bell_curve(candidates)
        utilities = normed_dist_utilities(voters, candidates)
        rankings = np.asarray(honest_rankings(utilities))
        trace = simulate_tvr_rounds(rankings, candidates)
        if trace is not None:
            return trial, voters, candidates, rankings, trace
    return None


# ── Rendering ─────────────────────────────────────────────────────────────────

def plot_wins(ax, wins, colors, labels, edgecolor='black', gap=0.15):
    """Head-to-head wins as stacked square blocks (shared style with IRV script)."""
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


def render_frame(
    voters,
    candidates,
    borda_scores,
    n_borda_active,
    colors,
    labels,
    frame_title,
    output_path,
    approval_pct,
    wins,
    eliminated=None,
    dark_background=True,
):
    """Render one TVR animation frame.

    borda_scores : ndarray of shape (n_cands,)
        Current Borda scores.  For candidates still active (including the loser
        currently being animated out), their scores may be non-zero.  For
        candidates eliminated in a previous round, scores are 0.
    n_borda_active : int
        The number of candidates whose Borda scores are non-trivially included in
        borda_scores (i.e. the active-set size when borda_scores was tallied).
        Used to convert Borda scores to avg_rank values for annotations.
    """
    if eliminated is None:
        eliminated = set()

    n_cands = len(candidates)
    # Candidates visible on scatter and coloured in bar charts (no loser, no old eliminated).
    remaining = [c for c in range(n_cands) if c not in eliminated]
    active_colors = [colors[n] if n not in eliminated else [0.5, 0.5, 0.5]
                     for n in range(n_cands)]

    bg, fg, grid, stroke_fg, legend_bg, legend_fg, voronoi_color = get_theme(dark_background)

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

    # ── Scatter plot ──────────────────────────────────────────────────────────
    voters_kwargs = {'marker': '.', 'alpha': 0.25, 's': 12}
    cands_kwargs = {'marker': 'o', 's': 30, 'edgecolors': fg}

    ax_sc.scatter([], [], color=fg, **voters_kwargs, label='Voters')
    ax_sc.scatter([], [], color=fg, **cands_kwargs, label='Candidates')
    ax_sc.legend(loc='lower right', numpoints=1, fontsize='small',
                 labelcolor=legend_fg, facecolor=legend_bg, edgecolor=legend_fg)
    setup_scatter_axis_sigma(ax_sc, voters)

    voronoi_plot_2d_axes(ax_sc, candidates[remaining], line_color=voronoi_color,
                         line_alpha=0.45)

    path_effects = [PathEffects.withStroke(linewidth=3, foreground=stroke_fg)]

    # Color voters by their first preference among remaining candidates.
    # Compute first-preference ballots from the original rankings for display only;
    # we pass 'ballots' directly as the IRV script does.  Here we color by which
    # remaining candidate each voter most prefers (closest).
    dists_voters = np.linalg.norm(
        voters[:, np.newaxis, :] - candidates[np.newaxis, remaining, :], axis=2
    )
    nearest_remaining_idx = np.argmin(dists_voters, axis=1)
    nearest_cand = np.array(remaining)[nearest_remaining_idx]

    for cand in remaining:
        cand_voters = voters[nearest_cand == cand]
        if len(cand_voters):
            ax_sc.scatter(cand_voters[:, 0], cand_voters[:, 1],
                          color=active_colors[cand], **voters_kwargs)

    ax_sc.scatter(candidates[remaining, 0], candidates[remaining, 1],
                  color=[active_colors[c] for c in remaining], **cands_kwargs)
    for cand in remaining:
        ax_sc.annotate(labels[cand], xy=candidates[cand], xytext=(0, -15),
                       textcoords='offset points', path_effects=path_effects,
                       color=fg)

    # ── Borda/average-rank bar chart ─────────────────────────────────────────
    # bar_height[c] = borda_scores[c] / n_voters, computed for ALL candidates.
    # This naturally equals 0 for candidates eliminated in a prior round
    # (their borda_scores[c] == 0), and decreases to 0 for the loser currently
    # being animated out.  The y-axis is fixed to [0, n_cands-1] throughout.
    # avg_rank for annotation: n_borda_active - borda/n_voters (1=best, n_borda_active=worst).
    n_total_voters = len(voters)
    bar_heights = np.array([max(0.0, borda_scores[c] / n_total_voters)
                            for c in range(n_cands)])
    avg_ranks = np.zeros(n_cands)
    for c in remaining:
        avg_ranks[c] = n_borda_active - borda_scores[c] / n_total_voters

    bars = ax_bar.bar(range(n_cands), bar_heights, tick_label=list(labels),
                      color=active_colors)
    for c, rect in enumerate(bars):
        height = rect.get_height()
        # Only annotate surviving candidates (not the loser being faded out).
        if height > 0 and c in remaining:
            ax_bar.annotate(
                f'{avg_ranks[c]:.1f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords='offset points',
                ha='center',
                va='bottom',
                color=fg,
            )

    ax_bar.set_ylim(0, n_cands - 1)
    # Custom y-ticks: rank labels (1 at top = n_cands-1, n_cands at bottom = 0)
    tick_vals = range(n_cands)
    tick_labels = [str(n_cands - v) for v in tick_vals]
    ax_bar.set_yticks(list(tick_vals))
    ax_bar.set_yticklabels(tick_labels)
    ax_bar.set_ylabel('Avg. rank (1=best)')
    ax_bar.grid(True, alpha=0.25, axis='y', color=grid)
    ax_bar.set_axisbelow(True)
    ax_bar.text(0.5, 1.04, frame_title, transform=ax_bar.transAxes,
                ha='center', va='center', color=fg)

    # ── Approval rating bar chart (identical to IRV) ─────────────────────────
    score_bars = ax_score.bar(range(n_cands), approval_pct, tick_label=list(labels),
                              color=active_colors)
    for rect in score_bars:
        height = rect.get_height()
        if height > 0:
            ax_score.annotate(
                f'{height:.0f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords='offset points',
                ha='center',
                va='bottom',
                color=fg,
            )
    ax_score.set_ylim(0, 100)
    ax_score.set_ylabel('Approval [%]')
    ax_score.grid(True, alpha=0.25, axis='y', color=grid)
    ax_score.set_axisbelow(True)
    ax_score.text(0.5, 1.04, 'Approval rating', transform=ax_score.transAxes,
                  ha='center', va='center', color=fg)

    # ── Head-to-head wins (identical to IRV) ─────────────────────────────────
    plot_wins(ax_wins, wins, active_colors, labels, edgecolor=fg, gap=0.1)
    ax_wins.text(0.5, 1.04, 'Head-to-head wins', transform=ax_wins.transAxes,
                 ha='center', va='center', color=fg)

    plt.tight_layout()
    plt.savefig(output_path, facecolor=bg, edgecolor='none')
    plt.close(fig)


# ── Election loading / search ──────────────────────────────────────────────────

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = Path('Images') / f'collapse_2d_tvr_{timestamp}_nc{n_cands}_nv{n_voters}'
output_dir.mkdir(parents=True, exist_ok=True)

colors = get_palette_colors(palette_name)
if not dark_background and palette_name == 'Set1_9' and len(colors) > 5:
    colors.pop(5)  # Yellow has low visibility on white backgrounds
if n_cands > len(colors):
    raise ValueError(
        f'n_cands={n_cands} exceeds palette "{palette_name}" size ({len(colors)}). '
        'Use fewer candidates or a larger palette.'
    )
colors = colors[:n_cands]
labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:n_cands]

if INPUT_POSITIONS is not None:
    data = np.load(INPUT_POSITIONS)
    voters = data['voters']
    candidates = data['candidates']
    utilities = normed_dist_utilities(voters, candidates)
    rankings = np.asarray(honest_rankings(utilities))
    trace = simulate_tvr_rounds(rankings, candidates)
    if trace is None:
        raise RuntimeError(
            f'TVR did not converge to center candidate for the election in '
            f'{INPUT_POSITIONS}.  Try a different positions.npz.'
        )
    trial = 0
    print(f'Loaded election from {INPUT_POSITIONS}.')
else:
    result = find_center_convergent_election(n_voters, n_cands, max_trials, disp=disp)
    if result is None:
        raise RuntimeError(
            'No TVR-convergent center election found. '
            'Increase max_trials or reduce n_cands.'
        )
    trial, voters, candidates, rankings, trace = result
    np.savez(output_dir / 'positions.npz', voters=voters, candidates=candidates)
    print(f'Found TVR-convergent election on trial {trial}.')

rounds = trace['rounds']
winner = trace['winner']

print('TVR elimination order:', ' -> '.join(candidate_name(r['loser']) for r in rounds))
print('TVR winner:', candidate_name(winner))

# Precompute constants used in every frame.
utilities = normed_dist_utilities(voters, candidates)
approval_pct = utilities.mean(axis=0) * 100
wins = count_wins(ranked_election_to_matrix(rankings))

# ── Animation ─────────────────────────────────────────────────────────────────

rng = np.random.default_rng()
frame = 0
eliminated = set()

# Initial frame: full Borda scores, no one eliminated.
initial_borda = rounds[0]['borda_before']
render_frame(
    voters=voters,
    candidates=candidates,
    borda_scores=initial_borda,
    n_borda_active=n_cands,
    colors=colors,
    labels=labels,
    frame_title='TVR start',
    output_path=output_dir / f'{frame:04d}.png',
    approval_pct=approval_pct,
    wins=wins,
    eliminated=set(),
    dark_background=dark_background,
)
frame += 1

for round_index, round_data in enumerate(rounds, start=1):
    loser = round_data['loser']
    eliminated_now = set(eliminated) | {loser}
    # n_borda_active = active-set size when borda_before was tallied (includes the loser).
    n_borda_active_this_round = n_cands - len(eliminated)

    # Elimination frame: loser grayed out but bar still shows their full Borda score.
    # Tallies are unchanged — transfer animation comes next.
    title = f'Round {round_index}: eliminate {candidate_name(loser)}'
    render_frame(
        voters=voters,
        candidates=candidates,
        borda_scores=round_data['borda_before'],
        n_borda_active=n_borda_active_this_round,
        colors=colors,
        labels=labels,
        frame_title=title,
        output_path=output_dir / f'{frame:04d}.png',
        approval_pct=approval_pct,
        wins=wins,
        eliminated=eliminated_now,
        dark_background=dark_background,
    )
    frame += 1

    # Transfer frames: remove loser from one voter's ballot at a time.
    # Their lower-ranked candidates gain +1 Borda; loser loses their contribution.
    all_voters = np.arange(len(voters))
    rng.shuffle(all_voters)
    per_frame = max(1, ceildiv(len(all_voters), frames_per_transfer))

    running_borda = round_data['borda_before'].copy()
    promoted_per_voter = round_data['promoted_per_voter']

    for step in range(frames_per_transfer):
        lo = step * per_frame
        hi = lo + per_frame
        batch = all_voters[lo:hi]

        for v in batch:
            # promoted[v] = active candidates voter ranked below the loser.
            # When loser is removed: each gains +1 Borda; loser loses len(promoted[v]).
            # (loser_pos_among_active = n_active - 1 - len(promoted[v]),
            #  so loser's Borda contribution from this voter = len(promoted[v]))
            running_borda[loser] -= len(promoted_per_voter[v])
            for c in promoted_per_voter[v]:
                running_borda[c] += 1.0

        render_frame(
            voters=voters,
            candidates=candidates,
            borda_scores=running_borda,
            n_borda_active=n_borda_active_this_round,
            colors=colors,
            labels=labels,
            frame_title=title,
            output_path=output_dir / f'{frame:04d}.png',
            approval_pct=approval_pct,
            wins=wins,
            eliminated=eliminated_now,
            dark_background=dark_background,
        )
        frame += 1

    eliminated.add(loser)

# Final frame: TVR winner.
final_borda = np.zeros(n_cands)
# winner's Borda score is trivially 0 (only candidate left; 0 points in 1-candidate round).
render_frame(
    voters=voters,
    candidates=candidates,
    borda_scores=final_borda,
    n_borda_active=1,
    colors=colors,
    labels=labels,
    frame_title=f'TVR winner: {candidate_name(winner)}',
    output_path=output_dir / f'{frame:04d}.png',
    approval_pct=approval_pct,
    wins=wins,
    eliminated=set(range(n_cands)) - {winner},
    dark_background=dark_background,
)

# ── Build GIF ─────────────────────────────────────────────────────────────────

frame_paths = sorted(output_dir.glob('*.png'))
images = [Image.open(p) for p in frame_paths]
gif_path = output_dir / 'collapse_2d_tvr.gif'
images[0].save(
    gif_path,
    save_all=True,
    append_images=images[1:],
    duration=50,
    loop=0,
)
for im in images:
    im.close()

print(f'Saved frames and GIF to {output_dir.resolve()}')
