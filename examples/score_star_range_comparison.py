"""
Compare Score and STAR winner distributions across ballot ranges.

Uses the same normalized spatial utilities as other 1D examples; only the
quantized score range changes. Ranges follow the discussion in
https://github.com/endolith/elsim/issues/20 (0–5 advocacy scale, 0–10 vs 1–10,
course-style 5 / 13 / 15 levels).

Similar layout to distributions_by_method.py (stacked histograms).
"""
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from seaborn import histplot, kdeplot

from elsim.elections import normal_electorate, normed_dist_utilities
from elsim.methods import score, star
from elsim.strategies import honest_normed_scores

n_elections = 50_000
n_voters = 1_000
n_cands = 7
dims = 1
disp = 0.5

ballot_specs = [
    ('0–5', 0, 5),
    ('0–10', 0, 10),
    ('1–10', 1, 10),
    ('Five grades (0–4)', 0, 4),
    ('13 levels (0–12)', 0, 12),
    ('15 levels (0–14)', 0, 14),
]

batch_size = 100
n_batches = n_elections // batch_size
assert n_batches * batch_size == n_elections


def human_format(num):
    for unit in ['', 'k', 'M', 'B', 'T']:
        if abs(num) < 1000:
            return f"{num:.3g}{unit}"
        num /= 1000.0


def simulate_batch(seed):
    rng = np.random.default_rng(seed)
    out = defaultdict(list)
    for _ in range(batch_size):
        v, c = normal_electorate(n_voters, n_cands, dims=dims, disp=disp,
                                 random_state=rng)
        utilities = normed_dist_utilities(v, c)
        for label, mn, mx in ballot_specs:
            ballots = honest_normed_scores(
                utilities, max_score=mx, min_score=mn)
            w_s = score(ballots, tiebreaker='random')
            w_t = star(ballots, tiebreaker='random')
            out[f'Score {label}'].append(c[w_s][0])
            out[f'STAR {label}'].append(c[w_t][0])
    return out


jobs = [delayed(simulate_batch)(k) for k in range(n_batches)]
print(f'{len(jobs)} tasks total:')
results = Parallel(n_jobs=-3, verbose=5)(jobs)

winners = {k: [v for d in results for v in d[k]] for k in results[0]}

title = (f'Score vs STAR by ballot range, {human_format(n_elections)} '
         f'1D elections, {human_format(n_voters)} voters, '
         f'{human_format(n_cands)} candidates, {disp} dispersion')

v, _ = normal_electorate(n_voters, 1000, dims=dims)

fig, ax = plt.subplots(nrows=len(winners), num=title, sharex=True,
                       constrained_layout=True, figsize=(7.5, 16))
fig.suptitle(title)
for n, method in enumerate(winners.keys()):
    histplot(winners[method], ax=ax[n], label='Winners', stat='density')
    ax[n].set_title(method, loc='left')
    ax[n].set_yticklabels([])
    ax[n].set_ylabel('')

    tmp = ax[n].twinx()
    kdeplot(v[:, 0], ax=tmp, ls=':', label='Voters')
    ax[n].plot([], [], ls=':', label='Voters')
    tmp.set_yticklabels([])
    tmp.set_ylabel('')

ax[0].set_xlim(-2.5, 2.5)
ax[0].legend()
