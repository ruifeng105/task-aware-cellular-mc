"""Per-history reliability constraint, oracle-gap closure and reliability-latency frontiers on B3.

Implements configs/b3_waiting_frontier_protocol.json (revision plan items 5
and 7), frozen before its only run. Every rule, fixed or adaptive, uses one
parameter per previous command and must reach the target for each history on
calibration samples; evaluation samples score success per history, the
fraction of the fixed-wait-to-oracle gap that feedback closes, and the
reliability-latency frontier. Also describes how task feasibility depends on
kappa and how the comparison depends on the failure penalty D. Simulation of
one published model structure, not evidence about measured cells.
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os

import numpy as np

import b3_readiness as br
import b3_waiting as bw
import b3_waiting_baselines as bl
import b3_waiting_robustness as rb

PROTOCOL = br.ROOT / 'configs/b3_waiting_frontier_protocol.json'
FREEZE = br.OUT / 'b3_waiting_frontier_protocol_freeze.json'
RESULTS = br.OUT / 'waiting_frontier_results.json'
THRESHOLDS = bl.FINE
TARGETS = (.90, .92, .94)
NOISES = (.005, .01)
PENALTIES = (140., 160., 200., 260.)
KAPPAS = (.30, .35, .40, .45, .50, .55, .60, .65, .70)
LEVELS = (.90, .92, .94)
ADAPTIVE = ('current', 'smoothed', 'history')
RULES = ADAPTIVE + ('fixed',)
PAIRS = (('history', 'fixed'), ('smoothed', 'fixed'), ('history', 'smoothed'))
PARTS = ('all',) + br.HISTORIES


def protocol_sha256():
    return hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()


def load_protocol():
    if protocol_sha256() != json.loads(FREEZE.read_text(encoding='utf-8'))['protocol_sha256']:
        raise SystemExit('Protocol differs from its freeze record; refusing to run.')
    bw.load_protocol()
    return json.loads(PROTOCOL.read_text(encoding='utf-8'))


def crossing_index(probability, thresholds):
    """First decision index at which the estimate reaches each threshold (else the last), shape (m, H, n)."""
    peak = np.maximum.accumulate(probability, axis=-1)
    hit = peak[None] >= np.asarray(thresholds)[:, None, None, None]
    return np.where(hit.any(-1), hit.argmax(-1), probability.shape[-1] - 1)


def scored(success, plants, k, deadline):
    """Per-plant success and failure-penalized completion for decision indices k (..., H, n)."""
    sub = np.broadcast_to(success[:, plants, :], k.shape + (success.shape[-1],))
    ok = np.take_along_axis(sub, k[..., None], -1)[..., 0]
    return ok, np.where(ok, br.GRID[k] + br.WINDOW_MIN, deadline)


def candidates(probability, success, plants, deadline):
    """Every parameter of every rule: (label, ok, completion) with arrays (H, n)."""
    out = {}
    for name in ('current', 'history'):
        ok, completion = scored(success, plants, crossing_index(probability[name], THRESHOLDS), deadline)
        out[name] = [(dict(threshold=float(c)), ok[i], completion[i]) for i, c in enumerate(THRESHOLDS)]
    out['smoothed'] = []
    for tau in bw.SMOOTH_TAUS:
        ok, completion = scored(success, plants, crossing_index(probability[f'smoothed_{tau:g}'], THRESHOLDS), deadline)
        out['smoothed'] += [(dict(tau_min=tau, threshold=float(c)), ok[i], completion[i]) for i, c in enumerate(THRESHOLDS)]
    k = np.broadcast_to(np.arange(br.GRID.size)[:, None, None], (br.GRID.size, success.shape[0], len(plants)))
    ok, completion = scored(success, plants, k, deadline)
    out['fixed'] = [(dict(wait_min=float(w)), ok[i], completion[i]) for i, w in enumerate(br.GRID)]
    return out


def tie_key(label):
    return tuple(label[key] for key in ('tau_min', 'threshold', 'wait_min') if key in label)


def choose(options, h, target):
    """Per-history choice: smallest calibration completion with success >= target (ties by tie_key)."""
    rows = [(label, float(ok[h].mean()), float(completion[h].mean())) for label, ok, completion in options]
    feasible = [row for row in rows if row[1] >= target]
    if feasible:
        label, s, c = min(feasible, key=lambda row: (row[2], tie_key(row[0])))
        return dict(label, target_reached=True, calibration=dict(success_rate=s, mean_completion_min=c))
    label, s, c = min(rows, key=lambda row: (-row[1], row[2], tie_key(row[0])))
    return dict(label, target_reached=False, calibration=dict(success_rate=s, mean_completion_min=c))


def lookup(options, choice):
    label = {k: v for k, v in choice.items() if k not in ('target_reached', 'calibration')}
    return next((ok, completion) for lab, ok, completion in options if lab == label)


def evaluated(options_ev, choices):
    """Per-plant outcomes on evaluation samples with one chosen parameter per history."""
    out = {}
    for name in RULES:
        parts = [lookup(options_ev[name], choices[name][hist]) for hist in br.HISTORIES]
        out[name] = (np.stack([parts[h][0][h] for h in range(len(parts))]),
                     np.stack([parts[h][1][h] for h in range(len(parts))]))
    return out


def gap_closure(rules):
    """(T_fixed - T_rule) / (T_fixed - T_oracle) per part for each adaptive rule."""
    out = {}
    for name in ADAPTIVE:
        out[name] = {}
        for part in PARTS:
            fixed, oracle = rules['fixed'][part]['mean_completion_min'], rules['oracle'][part]['mean_completion_min']
            span = fixed - oracle
            out[name][part] = (fixed - rules[name][part]['mean_completion_min']) / span if span > 0 else None
    return out


def paired(outcomes):
    out = {}
    for a, b in PAIRS:
        for part in PARTS:
            pick = (lambda x: x.mean(0)) if part == 'all' else (lambda x, h=br.HISTORIES.index(part): x[h])
            out[f'{a}-{b}/{part}'] = dict(success=pick(outcomes[a][0].astype(float)) - pick(outcomes[b][0].astype(float)),
                                          completion=pick(outcomes[a][1]) - pick(outcomes[b][1]))
    return out


def cell(options_cal, options_ev, oracle, target, with_bootstrap):
    choices = {name: {hist: choose(options_cal[name], h, target) for h, hist in enumerate(br.HISTORIES)} for name in RULES}
    outcomes = evaluated(options_ev, choices)
    rules = {name: rb.summarise(*pair) for name, pair in outcomes.items()}
    rules['oracle'] = rb.summarise(*oracle)
    differences = paired(outcomes)
    out = dict(choices=choices, rules=rules, gap_closure=gap_closure(rules),
               meets_per_history={name: bool(all(rules[name][h]['success_rate'] >= br.Q for h in br.HISTORIES))
                                  for name in RULES},
               paired={pair: {key: float(v.mean()) for key, v in d.items()} for pair, d in differences.items()})
    if with_bootstrap:
        out['bootstrap'] = rb.bootstrap(differences)
    return out


def oracle_outcomes(success, plants, deadline):
    k = crossing_index(success[:, plants, :].astype(float), [1.])[0]
    return scored(success, plants, k, deadline)


def frontier(options_ev, oracle):
    """Descriptive evaluation-side sweep per history: all points, non-dominated points and level minima."""
    out = {}
    for h, hist in enumerate(br.HISTORIES):
        block = {}
        for name in RULES:
            points = [dict(label, success_rate=float(ok[h].mean()), mean_completion_min=float(completion[h].mean()))
                      for label, ok, completion in options_ev[name]]
            front = [p for p in points
                     if not any(q['success_rate'] >= p['success_rate'] and q['mean_completion_min'] <= p['mean_completion_min']
                                and (q['success_rate'] > p['success_rate'] or q['mean_completion_min'] < p['mean_completion_min'])
                                for q in points)]
            levels = {}
            for level in LEVELS:
                feasible = [p for p in points if p['success_rate'] >= level]
                levels[f'{level:.2f}'] = min(feasible, key=lambda p: (p['mean_completion_min'], tie_key(p))) if feasible else None
            block[name] = dict(points=points, non_dominated=sorted(front, key=lambda p: p['mean_completion_min']), levels=levels)
        block['oracle'] = dict(success_rate=float(oracle[0][h].mean()), mean_completion_min=float(oracle[1][h].mean()))
        out[hist] = block
    return out


def run_repeat(r, noise):
    tables = dict(np.load(br.TABLES))
    success = br.success_table(tables)
    split_seed, noise_seed = rb.seeds(r)
    cal, ev = bw.split_samples(success.shape[1], seed=split_seed)
    observed = bw.noisy_observations(tables, noise, seed=noise_seed)
    on_cal = bw.estimator_probabilities(tables, success, noise, cal, cal, observed)
    on_ev = bw.estimator_probabilities(tables, success, noise, cal, ev, observed)
    result = dict(r=r, noise=noise, split_seed=split_seed, noise_seed=noise_seed, cells={}, penalty={})
    options_cal, options_ev = (candidates(on_cal, success, cal, br.DEADLINE_MIN),
                               candidates(on_ev, success, ev, br.DEADLINE_MIN))
    oracle = oracle_outcomes(success, ev, br.DEADLINE_MIN)
    for target in TARGETS:
        result['cells'][f'{target:.2f}'] = cell(options_cal, options_ev, oracle, target, r == 0)
    if r == 0:
        result['frontier'] = frontier(options_ev, oracle)
    if noise == .005:
        for deadline in PENALTIES:
            result['penalty'][f'{deadline:g}'] = cell(candidates(on_cal, success, cal, deadline),
                                                      candidates(on_ev, success, ev, deadline),
                                                      oracle_outcomes(success, ev, deadline), br.Q, False)
    return result


def _job(job):
    return run_repeat(*job)


def summarize_cells(rows):
    return dict(
        meets_per_history={name: int(sum(row['meets_per_history'][name] for row in rows)) for name in RULES},
        target_reached_on_calibration={name: int(sum(all(row['choices'][name][h]['target_reached'] for h in br.HISTORIES)
                                                     for row in rows)) for name in RULES},
        success={name: {part: rb.stats([row['rules'][name][part]['success_rate'] for row in rows]) for part in PARTS}
                 for name in RULES + ('oracle',)},
        completion={name: {part: rb.stats([row['rules'][name][part]['mean_completion_min'] for row in rows]) for part in PARTS}
                    for name in RULES + ('oracle',)},
        paired={pair: {key: rb.stats([row['paired'][pair][key] for row in rows]) for key in ('success', 'completion')}
                for pair in rows[0]['paired']},
        faster={pair: int(sum(row['paired'][pair]['completion'] < 0 for row in rows)) for pair in rows[0]['paired']},
        gap_closure={name: {part: rb.stats([row['gap_closure'][name][part] for row in rows]) for part in PARTS}
                     for name in ADAPTIVE})


def summarize(repeats):
    out = {}
    for noise in (f'{n:g}' for n in NOISES):
        rows = [repeats[str(r)][noise] for r in range(rb.REPEATS)]
        out[noise] = {target: summarize_cells([row['cells'][target] for row in rows]) for target in rows[0]['cells']}
    rows = [repeats[str(r)]['0.005'] for r in range(rb.REPEATS)]
    out['penalty'] = {d: summarize_cells([row['penalty'][d] for row in rows]) for d in rows[0]['penalty']}
    return out


def version3_gap_closure():
    """Re-expression of saved version-3 / robustness results against the pooled history-conditioned fixed waits."""
    robust = json.loads(rb.RESULTS.read_text(encoding='utf-8'))['repeats']
    conditioned = json.loads(bl.RESULTS.read_text(encoding='utf-8'))['repeats']['rows']
    rows = []
    for r in range(rb.REPEATS):
        rules = robust[str(r)]['0.005']['rules']
        fixed_c, single = conditioned[r]['evaluation']['all']['mean_completion_min'], rules['fixed']['all']['mean_completion_min']
        oracle = rules['oracle']['all']['mean_completion_min']
        rows.append({name: dict(vs_conditioned=(fixed_c - rules[name]['all']['mean_completion_min']) / (fixed_c - oracle),
                                vs_single=(single - rules[name]['all']['mean_completion_min']) / (single - oracle))
                     for name in ADAPTIVE})
    return dict(primary=rows[0], splits={name: {key: rb.stats([row[name][key] for row in rows]) for key in rows[0][name]}
                                         for name in ADAPTIVE})


def kappa_feasibility(tables):
    out = {}
    rise, naive = tables['rise'].astype(float), tables['naive_rise'].astype(float)[None, :, None]
    plants = np.arange(rise.shape[1])
    for kappa in KAPPAS:
        success = rise >= kappa * naive
        ok, completion = oracle_outcomes(success, plants, br.DEADLINE_MIN)
        pooled = success.mean(axis=(0, 1))
        hits = np.flatnonzero(pooled >= br.Q)
        out[f'{kappa:.2f}'] = dict(oracle=rb.summarise(ok, completion),
                                   first_fixed_wait_reaching_target=float(br.GRID[hits[0]]) if hits.size else None)
    return out


def main():
    load_protocol()
    tables = dict(np.load(br.TABLES))
    jobs = [(r, noise) for r in range(rb.REPEATS) for noise in NOISES]
    with ProcessPoolExecutor(max_workers=min(32, os.cpu_count() or 1)) as pool:
        results = list(pool.map(_job, jobs))
    repeats = {}
    for (r, noise), result in zip(jobs, results):
        repeats.setdefault(str(r), {})[f'{noise:g}'] = result
    result = dict(status='complete', protocol_sha256=protocol_sha256(),
                  frozen_utc=json.loads(FREEZE.read_text(encoding='utf-8'))['frozen_utc'],
                  repeats=repeats, summary=summarize(repeats), version3_gap_closure=version3_gap_closure(),
                  kappa_feasibility=kappa_feasibility(tables),
                  limitations=['Simulation of one published model structure; not evidence about measured cells.',
                               'Repeats redraw the split and observation noise only; one finite posterior set is reused.',
                               'Per-history targets are empirical calibration targets, not coverage guarantees.',
                               'Frontier points are swept on evaluation samples and are descriptive only.'])
    RESULTS.write_text(json.dumps(result, indent=2) + '\n', newline='\n')
    for noise in (f'{n:g}' for n in NOISES):
        for target, block in result['summary'][noise].items():
            print(noise, target, 'meets', block['meets_per_history'],
                  'faster', block['faster'],
                  'H-F', round(block['paired']['history-fixed/all']['completion']['median'], 2),
                  'S-F', round(block['paired']['smoothed-fixed/all']['completion']['median'], 2),
                  'G', {n: round(block['gap_closure'][n]['all']['median'], 3) for n in ADAPTIVE})
    print('v3', json.dumps(result['version3_gap_closure']['splits'], indent=1))
    print('kappa', {k: (round(v['oracle']['all']['success_rate'], 3), v['first_fixed_wait_reaching_target'])
                    for k, v in result['kappa_feasibility'].items()})


if __name__ == '__main__':
    main()
