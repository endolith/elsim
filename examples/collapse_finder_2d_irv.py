"""
Find a 2D IRV center-outward collapse and animate transfers by round.

This mirrors the transfer animation style used in elsim2k T2R examples, but
for full IRV: one candidate eliminated per round until two remain.
"""

from pathlib import Path

import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
from palettable.colorbrewer.qualitative import Set1_9 as cmap

from elsim.elections import normal_electorate, normed_dist_utilities
from elsim.methods._common import _inc_pointer, _tally_at_pointer
from elsim.strategies import honest_rankings


def candidate_name(candidate_index):
    """Convert candidate index to name (A, B, C, etc.)."""
    return chr(65 + candidate_index)


def ceildiv(a, b):
    """Ceiling division for positive integers."""
    return -(-a // b)


def simulate_irv_rounds(rankings, candidates):
    """
    Simulate IRV with round-by-round trace data.

    Uses elsim3k's numba-compiled _tally_at_pointer and _inc_pointer for speed.
    Returns None if there is any tie for elimination, because this example is
    intended to find a clean center-outward pattern with deterministic rounds.
    """
    election = np.asarray(rankings, dtype=np.intp)
    n_voters, n_cands = election.shape
    dists_to_origin = np.linalg.norm(candidates, axis=1)

    pointer = np.zeros(n_voters, dtype=np.uint8)
    first_tallies = np.empty(n_cands, dtype=np.uint64)
    active = set(range(n_cands))
    eliminated = set()
    rounds = []

    while len(active) > 2:
        _tally_at_pointer(first_tallies, election, pointer)
        tallies = first_tallies.copy()
        active_sorted = sorted(active)

        min_tally = min(tallies[cand] for cand in active_sorted)
        low_scorers = [cand for cand in active_sorted if tallies[cand] == min_tally]
        if len(low_scorers) != 1:
            return None

        loser = low_scorers[0]

        # Strict condition: eliminate closest remaining candidate every round.
        expected_loser = min(active_sorted, key=lambda cand: (dists_to_origin[cand], cand))
        if loser != expected_loser:
            return None

        ballots_before = election[np.arange(n_voters), pointer].copy()
        affected_voters = np.flatnonzero(ballots_before == loser)
        next_eliminated = eliminated | {loser}

        _inc_pointer(election, pointer, next_eliminated)
        ballots_after = election[np.arange(n_voters), pointer].copy()
        _tally_at_pointer(first_tallies, election, pointer)
        tallies_after = first_tallies.copy()

        rounds.append({
            'loser': loser,
            'ballots_before': ballots_before,
            'ballots_after': ballots_after,
            'tallies_before': tallies,
            'tallies_after': tallies_after,
            'affected_voters': affected_voters,
        })

        active.remove(loser)
        eliminated = next_eliminated

    final_two = sorted(active)
    farthest_two = sorted(np.argsort(dists_to_origin)[-2:])
    if final_two != farthest_two:
        return None

    ballots_final = election[np.arange(n_voters), pointer].copy()
    _tally_at_pointer(first_tallies, election, pointer)

    return {
        'rounds': rounds,
        'final_two': final_two,
        'final_ballots': ballots_final,
        'final_tallies': first_tallies.copy(),
        'dists_to_origin': dists_to_origin,
    }


def find_center_outward_election(n_voters, n_cands, max_trials, disp=1.0):
    """Sample random 2D elections until the strict center-outward pattern appears."""
    for trial in range(1, max_trials + 1):
        voters, candidates = normal_electorate(n_voters, n_cands, dims=2, disp=disp)
        utilities = normed_dist_utilities(voters, candidates)
        rankings = np.asarray(honest_rankings(utilities))
        trace = simulate_irv_rounds(rankings, candidates)
        if trace is not None:
            return trial, voters, candidates, trace
    return None


def render_frame(
    voters,
    candidates,
    ballots,
    tallies,
    colors,
    labels,
    frame_title,
    output_path,
    eliminated=None,
):
    """Render one animation frame in the same visual style as T2R examples."""
    if eliminated is None:
        eliminated = set()

    n_cands = len(candidates)
    n_voters = len(voters)
    active_colors = [colors[n] if n not in eliminated else [0.8, 0.8, 0.8] for n in range(n_cands)]

    fig = plt.figure(figsize=(9, 7.5))
    ax_sc = plt.subplot2grid(shape=(4, 3), loc=(0, 0), colspan=2, rowspan=4)
    ax_bar = plt.subplot2grid(shape=(4, 3), loc=(1, 2), rowspan=2)

    voters_kwargs = {'marker': '.', 'alpha': 0.25, 's': 5}
    cands_kwargs = {'marker': 'o', 's': 30, 'edgecolors': 'white'}

    ax_sc.scatter([], [], color='k', **voters_kwargs, label='Voters')
    ax_sc.scatter([], [], color='k', **cands_kwargs, label='Candidates')
    ax_sc.legend(loc='lower right', numpoints=1, fontsize='small')
    ax_sc.grid(True, alpha=0.2)
    ax_sc.set_axisbelow(True)
    ax_sc.axis('square')
    ax_sc.axis([-3, 3, -3, 3])

    path_effects = [PathEffects.withStroke(linewidth=3, foreground='w')]

    for cand in range(n_cands):
        cand_voters = voters[ballots == cand]
        if len(cand_voters):
            ax_sc.scatter(cand_voters[:, 0], cand_voters[:, 1], color=active_colors[cand], **voters_kwargs)

    ax_sc.scatter(candidates[:, 0], candidates[:, 1], color=active_colors, **cands_kwargs)
    for cand, pos in enumerate(candidates):
        ax_sc.annotate(labels[cand], xy=pos, xytext=(0, -15), textcoords='offset points', path_effects=path_effects)

    bars = ax_bar.bar(range(n_cands), tallies / n_voters * 100, tick_label=list(labels), color=active_colors)
    for rect in bars:
        height = rect.get_height()
        if height > 0:
            ax_bar.annotate(
                f'{height:.0f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords='offset points',
                ha='center',
                va='bottom',
            )

    ax_bar.set_ylim(0, 100)
    ax_bar.set_ylabel('Votes [%]')
    ax_bar.grid(True, alpha=0.25, axis='y')
    ax_bar.set_axisbelow(True)
    ax_bar.text(0.5, 1.04, frame_title, transform=ax_bar.transAxes, ha='center', va='center')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


n_voters = 5000
n_cands = 6
max_trials = 100_000
frames_per_transfer = 60
output_dir = Path('Images') / 'collapse_2d_irv'
output_dir.mkdir(parents=True, exist_ok=True)

colors = list(cmap.mpl_colors)
assert cmap.name == 'Set1'
if len(colors) > 5:
    colors.pop(5)  # Throw away yellow; difficult to see on white backgrounds.

if n_cands > len(colors):
    raise ValueError(f'n_cands={n_cands} exceeds available palette size={len(colors)}')

colors = colors[:n_cands]
labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:n_cands]

result = find_center_outward_election(n_voters, n_cands, max_trials)
if result is None:
    raise RuntimeError('No strict center-outward collapse found. Increase max_trials or reduce n_cands.')

trial, voters, candidates, trace = result
rounds = trace['rounds']
final_two = trace['final_two']

print(f'Found strict center-outward IRV collapse on trial {trial}.')
print('Elimination order:', ' -> '.join(candidate_name(r['loser']) for r in rounds))
print('Final two:', candidate_name(final_two[0]), candidate_name(final_two[1]))

raise SystemExit

frame = 0
initial = rounds[0]
render_frame(
    voters=voters,
    candidates=candidates,
    ballots=initial['ballots_before'],
    tallies=initial['tallies_before'],
    colors=colors,
    labels=labels,
    frame_title='IRV start',
    output_path=output_dir / f'{frame:04d}.png',
    eliminated=set(),
)
frame += 1

rng = np.random.default_rng()
eliminated = set()

for round_index, round_data in enumerate(rounds, start=1):
    loser = round_data['loser']
    ballots = round_data['ballots_before'].copy()
    eliminated_now = set(eliminated)
    eliminated_now.add(loser)

    affected = round_data['affected_voters'].copy()
    rng.shuffle(affected)
    per_frame = max(1, ceildiv(len(affected), frames_per_transfer))

    for step in range(frames_per_transfer + 1):
        lo = step * per_frame
        hi = lo + per_frame
        changing = affected[lo:hi]
        ballots[changing] = round_data['ballots_after'][changing]
        tallies = np.bincount(ballots, minlength=n_cands)

        title = f'Round {round_index}: eliminate {candidate_name(loser)}'
        render_frame(
            voters=voters,
            candidates=candidates,
            ballots=ballots,
            tallies=tallies,
            colors=colors,
            labels=labels,
            frame_title=title,
            output_path=output_dir / f'{frame:04d}.png',
            eliminated=eliminated_now,
        )
        frame += 1

    eliminated.add(loser)

render_frame(
    voters=voters,
    candidates=candidates,
    ballots=trace['final_ballots'],
    tallies=trace['final_tallies'],
    colors=colors,
    labels=labels,
    frame_title=f'Final two: {candidate_name(final_two[0])} vs {candidate_name(final_two[1])}',
    output_path=output_dir / f'{frame:04d}.png',
    eliminated=set(range(n_cands)) - set(final_two),
)

print(f'Saved frames to {output_dir.resolve()}')
