"""In-silico task readiness and adaptive waiting on the authors' B3 model.

Implements configs/b3_readiness_protocol_v2.json. Version 1 (task 0.8,
baseline-band reset) was run once and was infeasible for every sample; its
outcome is kept in results/b3/readiness_results_v1.json. This simulates one
published model structure and is not evidence about measured cells.
Usage: python b3_readiness.py [--reuse-tables]
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import b3_model as b3

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'configs/b3_readiness_protocol_v2.json'
SIMULATION_PROTOCOL = ROOT / 'configs/b3_readiness_protocol.json'  # v1 holds the unchanged simulation settings
OUT = ROOT / 'results/b3'
FREEZE = OUT / 'b3_readiness_protocol_v2_freeze.json'
RESULTS = OUT / 'readiness_results.json'
DECISIONS = OUT / 'readiness_decisions.csv'
TABLES = OUT / 'readiness_tables.npz'
HISTORIES = ('short', 'long')
GRID = np.arange(0., 121., 2.)
# Settings as frozen in the protocols (version 2 changes only KAPPA, the reset rule and S1).
KAPPA, WINDOW_MIN, PROBE_MIN, DEADLINE_MIN = .5, 20., 5., 140.
NOISE_SD, KERNEL_SD, SEED, Q = .005, .005, 20261005, .9
FIXED_WAITS = (6., 12., 24., 48.)
REPORTER_BINS = np.round(np.arange(.030, .0701, .005), 3)


def protocol_sha256():
    return hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()


def load_protocol():
    """The frozen simulation settings (v1) after checking the v2 freeze record."""
    if protocol_sha256() != json.loads(FREEZE.read_text(encoding='utf-8'))['protocol_sha256']:
        raise SystemExit('Protocol differs from its freeze record; refusing to run.')
    if json.loads((OUT / 'verification_results.json').read_text(encoding='utf-8'))['status'] != 'passed':
        raise SystemExit('B3 acceptance check has not passed.')
    return json.loads(SIMULATION_PROTOCOL.read_text(encoding='utf-8'))


def probe_rise(model, row, history, concentration, wait):
    """Incremental FRET rise within the task window after a probe at `wait`."""
    times = np.arange(wait, wait + WINDOW_MIN + 1., 1.)
    states = model.simulate_schedule_states(row, list(history) + [(wait, wait + PROBE_MIN)], concentration, times)
    fret = model.fret_from_states(row, states)
    return float(fret.max() - fret[0])


def _job(job):
    kind, row, history, concentration, wait = job
    model, row = b3.worker_model(), np.array(row)
    if kind == 'history':
        return model.fret_from_states(row, model.simulate_schedule_states(row, list(history), concentration, GRID))
    return probe_rise(model, row, history, concentration, wait)


def simulate_tables(protocol, workers=None):
    posterior, dose = b3.load_posterior(), protocol['command_ng_per_ml']
    histories = {name: tuple(tuple(s) for s in protocol['histories'][name]) for name in HISTORIES}
    jobs = [('naive', tuple(r), (), dose, 0.) for r in posterior]
    jobs += [('history', tuple(r), histories[n], dose, 0.) for n in HISTORIES for r in posterior]
    jobs += [('probe', tuple(r), histories[n], dose, float(w)) for n in HISTORIES for r in posterior for w in GRID]
    with ProcessPoolExecutor(max_workers=workers or min(32, os.cpu_count() or 1)) as pool:
        out = list(pool.map(_job, jobs, chunksize=256))
    samples, waits = len(posterior), len(GRID)
    return dict(naive_rise=np.array(out[:samples], dtype=np.float32),
                history_fret=np.array(out[samples:3 * samples], dtype=np.float32).reshape(2, samples, waits),
                rise=np.array(out[3 * samples:], dtype=np.float32).reshape(2, samples, waits))


def success_table(tables):
    return tables['rise'] >= KAPPA * tables['naive_rise'][None, :, None]


def first_crossing(probability, q):
    hits = np.flatnonzero(probability >= q)
    return int(hits[0]) if hits.size else len(probability) - 1


def half_decay_index(observed):
    """First time the observed rise above baseline is at most half its running maximum."""
    rise = observed - 1.
    hits = np.flatnonzero(rise <= .5 * np.maximum.accumulate(rise))
    return int(hits[0]) if hits.size else len(observed) - 1


def history_probability(observed, trajectories, success, plant, sigma):
    """Success probability per decision time from the other samples, weighted by
    every observation up to that time under the known command history."""
    others = np.arange(len(trajectories)) != plant
    log_w = -np.cumsum((trajectories[others] - observed[None, :]) ** 2, axis=1) / (2 * sigma ** 2)
    log_w -= log_w.max(axis=0, keepdims=True)
    weights = np.exp(log_w)
    return (weights / weights.sum(axis=0, keepdims=True) * success[others]).sum(0)


class CurrentReadiness:
    """Success probability given only the current observation, pooling every
    (sample, history, wait) state of the other samples with a Gaussian kernel."""

    def __init__(self, fret, success, sigma, points=4001):
        self.fret, self.success, self.sigma = fret.astype(float), success.astype(float), sigma
        pooled_f, pooled_s = self.fret.ravel(), self.success.ravel()
        self.grid = np.linspace(pooled_f.min() - 6 * sigma, pooled_f.max() + 6 * sigma, points)
        self.s0, self.s1 = np.zeros(points), np.zeros(points)
        for chunk in np.array_split(np.arange(points), 80):
            kernel = np.exp(-(self.grid[chunk, None] - pooled_f[None, :]) ** 2 / (2 * sigma ** 2))
            self.s0[chunk], self.s1[chunk] = kernel.sum(1), kernel @ pooled_s

    def probability(self, observed, plant):
        own_f, own_s = self.fret[:, plant, :].ravel(), self.success[:, plant, :].ravel()
        own = np.exp(-(observed[:, None] - own_f[None, :]) ** 2 / (2 * self.sigma ** 2))
        s0 = np.interp(observed, self.grid, self.s0) - own.sum(1)
        s1 = np.interp(observed, self.grid, self.s1) - own @ own_s
        return np.where(s0 > 1e-9, s1 / np.maximum(s0, 1e-300), 0.)


def observations(tables):
    fret = tables['history_fret'].astype(float)
    observed = np.empty_like(fret)
    for h in range(fret.shape[0]):
        for plant in range(fret.shape[1]):
            rng = np.random.default_rng(SEED + 2 * plant + h)
            observed[h, plant] = fret[h, plant] + rng.normal(0., NOISE_SD, GRID.size)
    return observed


def decide_all(tables):
    success_all = success_table(tables)
    observed, rows = observations(tables), []
    current = CurrentReadiness(tables['history_fret'], success_all, KERNEL_SD)
    for h, name in enumerate(HISTORIES):
        trajectories, success = tables['history_fret'][h].astype(float), success_all[h]
        for plant in range(trajectories.shape[0]):
            y = observed[h, plant]
            choice = {f'fixed_{w:g}': int(np.flatnonzero(GRID == w)[0]) for w in FIXED_WAITS}
            choice['threshold_reset'] = half_decay_index(y)
            choice['readiness_current'] = first_crossing(current.probability(y, plant), Q)
            choice['readiness_history'] = first_crossing(history_probability(y, trajectories, success, plant, KERNEL_SD), Q)
            choice['oracle'] = first_crossing(success[plant].astype(float), 1.)
            for policy, k in choice.items():
                ok = bool(success[plant, k])
                rows.append(dict(history=name, plant=plant, policy=policy, wait_min=float(GRID[k]), success=int(ok),
                                 completion_min=float(GRID[k] + WINDOW_MIN) if ok else DEADLINE_MIN))
    return pd.DataFrame(rows)


def matched_reporter_readiness(tables, success):
    """S1: readiness per reporter bin and history, and the weighted long-minus-short difference."""
    reporter = tables['history_fret'].astype(float) - 1.
    bins, weighted, weight = [], 0., 0
    for lo, hi in zip(REPORTER_BINS[:-1], REPORTER_BINS[1:]):
        row = dict(reporter_rise_from=float(lo), reporter_rise_to=float(hi))
        for h, name in enumerate(HISTORIES):
            mask = (reporter[h] >= lo) & (reporter[h] < hi)
            row[name] = dict(states=int(mask.sum()), readiness=float(success[h][mask].mean()) if mask.any() else None)
        if row['short']['states'] and row['long']['states']:
            n = row['short']['states'] + row['long']['states']
            weighted += n * (row['long']['readiness'] - row['short']['readiness'])
            weight += n
        bins.append(row)
    return dict(bins=bins, weighted_mean_difference_long_minus_short=weighted / weight if weight else None,
                states_in_common_bins=weight)


def summarize(tables, decisions):
    success = success_table(tables)
    policies = {}
    for policy, group in decisions.groupby('policy', sort=False):
        policies[policy] = dict(success_rate=float(group.success.mean()), mean_completion_min=float(group.completion_min.mean()),
                                mean_wait_min=float(group.wait_min.mean()),
                                by_history={n: dict(success_rate=float(g.success.mean()),
                                                    mean_completion_min=float(g.completion_min.mean()),
                                                    mean_wait_min=float(g.wait_min.mean()))
                                            for n, g in group.groupby('history', sort=False)})
    curve = [dict(wait_min=float(w), success_rate=float(success[:, :, k].mean()),
                  mean_completion_min=float(np.where(success[:, :, k], w + WINDOW_MIN, DEADLINE_MIN).mean()))
             for k, w in enumerate(GRID)]
    matched = {}
    for policy in ('threshold_reset', 'readiness_current', 'readiness_history', 'oracle'):
        rate = policies[policy]['success_rate']
        fixed = next((point for point in curve if point['success_rate'] >= rate), None)
        matched[policy] = dict(policy_success_rate=rate, policy_mean_completion_min=policies[policy]['mean_completion_min'],
                               matching_fixed_wait=fixed)
    ratio = tables['rise'].astype(float) / tables['naive_rise'].astype(float)[None, :, None]
    return dict(policies=policies, fixed_wait_curve=curve, matched_reliability=matched,
                readiness_curves=dict(waits_min=GRID.tolist(), **{n: success[h].mean(0).tolist() for h, n in enumerate(HISTORIES)}),
                matched_reporter_readiness=matched_reporter_readiness(tables, success),
                version1_check=dict(max_recovered_fraction=float(ratio.max()),
                                    states_within_0_01_of_baseline=int((np.abs(tables['history_fret'].astype(float) - 1.) <= .01).sum())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-tables', action='store_true', help='reuse saved simulation tables')
    args = parser.parse_args()
    protocol = load_protocol()
    if args.reuse_tables and TABLES.exists():
        tables = {k: v for k, v in np.load(TABLES).items() if k in ('naive_rise', 'history_fret', 'rise')}
    else:
        tables = simulate_tables(protocol)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(TABLES, **tables)
    tables = dict(np.load(TABLES))
    decisions = decide_all(tables)
    decisions.to_csv(DECISIONS, index=False, lineterminator='\n')
    summary = summarize(tables, decisions)
    pol, matched = summary['policies'], summary['matched_reliability']
    fixed = matched['readiness_history']['matching_fixed_wait']
    statements = dict(
        S1_matched_reporter=summary['matched_reporter_readiness']['weighted_mean_difference_long_minus_short'],
        S2_history_policy_vs_matching_fixed_wait=dict(
            policy=dict(success_rate=pol['readiness_history']['success_rate'], mean_completion_min=pol['readiness_history']['mean_completion_min']),
            matching_fixed_wait=fixed, shorter=None if fixed is None else pol['readiness_history']['mean_completion_min'] < fixed['mean_completion_min']),
        S3_history_vs_current=dict(history=pol['readiness_history'], current=pol['readiness_current']))
    result = dict(status='complete', protocol_sha256=protocol_sha256(),
                  frozen_utc=json.loads(FREEZE.read_text(encoding='utf-8'))['frozen_utc'],
                  n_samples=int(tables['naive_rise'].size), statements=statements, **summary,
                  limitations=['Simulation of one published model structure; not evidence about measured cells.',
                               'Posterior samples are population-mean uncertainty, not cell heterogeneity.',
                               'Controllers share the plant structure (leave-one-out samples): no structural mismatch.',
                               'Task band and windows are illustrative, not an application calibration; version 2 followed a degenerate version 1.'])
    RESULTS.write_text(json.dumps(result, indent=2) + '\n', newline='\n')
    print(json.dumps(dict(S1=statements['S1_matched_reporter'],
                          policies={p: {k: round(v, 3) for k, v in r.items() if k != 'by_history'} for p, r in pol.items()},
                          matched={p: (m['matching_fixed_wait'] or {}) for p, m in matched.items()}), indent=2))


if __name__ == '__main__':
    main()
