"""Declarative Monte Carlo studies over scenario grids."""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

import numpy as np

from elsim.methods import condorcet
from elsim.strategies import honest_normed_scores

from .metrics import normalize_utilities, rankings_from


def _default_score_rule(utilities):
    return honest_normed_scores(utilities, 5)


def _method_winner(method, ballots, tiebreaker):
    if tiebreaker is None:
        return method(ballots)
    return method(ballots, tiebreaker=tiebreaker)


@dataclass
class StudyResult:
    """Values for one metric, one row per scenario."""

    scenarios: list[dict[str, Any]]
    values: list[dict[str, float]]
    method_names: list[str]
    metric: str
    n_elections: int

    def by_method(self) -> dict[str, list[float]]:
        """Method name -> values in scenario order."""
        return {name: [v[name] for v in self.values]
                for name in self.method_names}

    def table(self, methods=None, *, percent: bool = True) -> list[list]:
        """Method rows ready for ``tabulate``; scenario columns."""
        names = list(methods) if methods is not None else self.method_names
        out = []
        for name in names:
            values = [v[name] for v in self.values]
            if self.metric == 'condorcet' and percent:
                denominator = ([v['CW'] for v in self.values]
                               if name != 'CW' else
                               [self.n_elections] * len(self.values))
                values = [100 * value / den if den else np.nan
                          for value, den in zip(values, denominator,
                                                strict=True)]
            out.append([name, *values])
        return out


class Study:
    """
    Run one metric over a list of scenarios, optionally in parallel.

    Parameters
    ----------
    scenarios : list of dict
        Scenario dictionaries, e.g. from :func:`elsim.studies.grid` or
        :func:`elsim.studies.conditions`.
    electorate : model
        Object with ``draw(scenario, rng)`` and
        ``draw_set(scenario, n, rng)``.
    n_elections : int
        Total number of simulated electorates.
    ranked_methods, rated_methods, scored_methods : mapping, optional
        Name -> callable maps.  Ranked methods receive ballots, rated
        methods receive utilities, scored methods receive score ballots.
    metric : {'condorcet', 'sue'}
        ``condorcet`` counts Condorcet agreement; ``sue`` computes
        per-candidate-set social-utility efficiency.
    electorates_per_set : int, optional
        Group electorates per candidate set.  The default 1 gives one
        candidate set per electorate.  Merrill's spatial design uses 5.
    n_jobs : int, optional
        ``1`` serial, ``-1`` all CPUs, positive values that many workers.
        Requires the optional ``joblib`` dependency for values other than 1.
    normalization : {'standard', 'range', None}, optional
        Utility normalization used for SUE (ignored for Condorcet).
    tiebreaker : {None, 'random', 'order'}, optional
        Passed through to each voting method.
    random_state : int, Generator or None, optional
        Seed (or generator) for reproducible scenario/chunk seeds.
    """

    def __init__(
        self,
        scenarios,
        electorate,
        n_elections: int,
        *,
        ranked_methods=None,
        rated_methods=None,
        scored_methods=None,
        score_rule: Callable | None = None,
        metric: str = 'condorcet',
        electorates_per_set: int = 1,
        n_jobs: int = 1,
        normalization: str | None = 'standard',
        tiebreaker: str | None = 'random',
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        self.scenarios = [dict(s) for s in scenarios]
        self.electorate = electorate
        self.n_elections = int(n_elections)
        self.ranked_methods = dict(ranked_methods or {})
        self.rated_methods = dict(rated_methods or {})
        self.scored_methods = dict(scored_methods or {})
        self.score_rule = score_rule or _default_score_rule
        if metric not in ('condorcet', 'sue'):
            raise ValueError("metric must be 'condorcet' or 'sue'")
        if electorates_per_set < 1:
            raise ValueError('electorates_per_set must be positive')
        if self.n_elections % electorates_per_set:
            raise ValueError('n_elections must be divisible by '
                             'electorates_per_set')
        self.metric = metric
        self.electorates_per_set = int(electorates_per_set)
        self.n_jobs = int(n_jobs)
        self.normalization = normalization
        self.tiebreaker = tiebreaker
        self.random_state = random_state
        self.method_names = [*self.ranked_methods, *self.rated_methods,
                             *self.scored_methods]
        if metric == 'condorcet':
            self.method_names.append('CW')

    @property
    def n_sets(self) -> int:
        return self.n_elections // self.electorates_per_set

    def _seed_sequence(self) -> np.random.SeedSequence:
        state = self.random_state
        if isinstance(state, np.random.Generator):
            state = int(state.integers(0, 2**32))
        return np.random.SeedSequence(state)

    def _tasks(self):
        n_sets = self.n_sets
        workers = (os.cpu_count() or 1) if self.n_jobs == -1 else self.n_jobs
        workers = max(1, workers)
        chunks = min(n_sets, max(1, workers // max(1, len(self.scenarios))))
        if chunks == 1:
            tasks = [(i, 0, n_sets) for i in range(len(self.scenarios))]
        else:
            tasks = []
            for i, _scenario in enumerate(self.scenarios):
                bounds = np.linspace(0, n_sets, chunks + 1, dtype=int)
                tasks.extend((i, int(a), int(b))
                             for a, b in zip(bounds[:-1], bounds[1:],
                                             strict=True))
        seeds = self._seed_sequence().spawn(len(tasks))
        return [(i, self, self.scenarios[i], seed, start, stop)
                for (i, start, stop), seed in zip(tasks, seeds, strict=True)]

    def run(self) -> StudyResult:
        tasks = self._tasks()
        if self.n_jobs == 1 or len(tasks) == 1:
            partials = [_run_chunk(*task[1:]) for task in tasks]
        else:
            try:
                from joblib import Parallel, delayed
            except ImportError as exc:
                raise ImportError(
                    'parallel studies require joblib; install '
                    "'elsim[parallel]' or set n_jobs=1") from exc
            partials = Parallel(n_jobs=self.n_jobs)(
                delayed(_run_chunk)(*task[1:]) for task in tasks)

        values = [
            self._merge([p for p, task in zip(partials, tasks, strict=True)
                         if task[0] == scenario_index])
            for scenario_index in range(len(self.scenarios))
        ]
        return StudyResult(self.scenarios, values, self.method_names,
                           self.metric, self.n_elections)

    def _merge(self, partials):
        if self.metric == 'condorcet':
            total = Counter()
            for partial in partials:
                total.update(partial)
            return {name: float(total.get(name, 0))
                    for name in self.method_names}
        total = {name: 0.0 for name in self.method_names}
        count = 0
        for partial, n_sets in partials:
            for name in self.method_names:
                total[name] += partial[name]
            count += n_sets
        return {name: total[name] / count for name in self.method_names}


def _election_ballots(election, study: Study):
    utilities = election.utilities
    if utilities is None and (study.rated_methods or study.scored_methods
                              or study.metric == 'sue'):
        raise ValueError('this study needs utilities but the model returned '
                         'rankings only')

    rankings = election.rankings
    if rankings is None:
        rankings = rankings_from(utilities)

    scores = None
    if study.scored_methods:
        scores = study.score_rule(utilities)

    work_utilities = utilities
    if utilities is not None:
        work_utilities = normalize_utilities(utilities,
                                             study.normalization)
    return work_utilities, rankings, scores


def _winners(study: Study, work_utilities, rankings, scores):
    out = {}
    for name, method in study.ranked_methods.items():
        out[name] = _method_winner(method, rankings, study.tiebreaker)
    for name, method in study.rated_methods.items():
        out[name] = _method_winner(method, work_utilities, study.tiebreaker)
    for name, method in study.scored_methods.items():
        out[name] = _method_winner(method, scores, study.tiebreaker)
    return out


def _run_chunk(study: Study, scenario, seed, set_start, set_stop):
    rng = np.random.default_rng(seed)
    if study.metric == 'condorcet':
        counts = Counter()
        for _ in range(set_start * study.electorates_per_set,
                       set_stop * study.electorates_per_set):
            election = study.electorate.draw(scenario, rng)
            utilities, rankings, scores = _election_ballots(election, study)
            cw = condorcet(rankings)
            if cw is None:
                continue
            counts['CW'] += 1
            for name, winner in _winners(study, utilities, rankings,
                                         scores).items():
                if winner == cw:
                    counts[name] += 1
        return counts

    sums = {name: 0.0 for name in study.method_names}
    n_sets = 0
    for _ in range(set_start, set_stop):
        elections = study.electorate.draw_set(
            scenario, study.electorates_per_set, rng)
        utility_sums = {name: 0.0 for name in study.method_names}
        max_utility_sum = 0.0
        random_utility_sum = 0.0
        for election in elections:
            utilities, rankings, scores = _election_ballots(election, study)
            utility_sum = utilities.sum(axis=0)
            max_utility_sum += float(utility_sum.max())
            random_utility_sum += float(utility_sum.mean())
            for name, winner in _winners(study, utilities, rankings,
                                         scores).items():
                utility_sums[name] += float(utility_sum[winner])
        denominator = max_utility_sum - random_utility_sum
        for name in study.method_names:
            sums[name] += ((utility_sums[name] - random_utility_sum)
                           / denominator * 100)
        n_sets += 1
    return sums, n_sets


def run(*args, **kwargs) -> StudyResult:
    """Shortcut for ``Study(*args, **kwargs).run()``."""
    return Study(*args, **kwargs).run()
