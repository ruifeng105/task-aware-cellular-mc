"""Shared calibration margin for the calibrated waiting rules on B3.

Implements configs/b3_waiting_margin_protocol.json (review items P0-1 and
P0-2), frozen before its only run. Every rule, fixed or adaptive, is
calibrated to the same target 0.9 + epsilon and evaluated against 0.9 on the
20 splits of the robustness study, so a fixed comparator sitting at the edge
of the target cannot drive the comparison. Simulation of one published model
structure, not evidence about measured cells.
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

PROTOCOL = br.ROOT / 'configs/b3_waiting_margin_protocol.json'
FREEZE = br.OUT / 'b3_waiting_margin_protocol_freeze.json'
RESULTS = br.OUT / 'waiting_margin_results.json'
TARGETS = (.90, .92, .94)
NOISES = (.005, .01)
GRIDS = {'coarse': np.array(bw.THRESHOLDS), 'fine': bl.FINE}
PAIRS = (('history', 'conditioned'), ('smoothed', 'conditioned'), ('history', 'smoothed'), ('history', 'fixed'))
RULES = ('current', 'smoothed', 'history', 'fixed', 'conditioned', 'conditioned_per_history')


def protocol_sha256():
    return hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()


def load_protocol():
    if protocol_sha256() != json.loads(FREEZE.read_text(encoding='utf-8'))['protocol_sha256']:
        raise SystemExit('Protocol differs from its freeze record; refusing to run.')
    bw.load_protocol()
    return json.loads(PROTOCOL.read_text(encoding='utf-8'))


def first_feasible(probability, success, plants, grid, target):
    for threshold in grid:
        outcome = bw.outcomes(probability, success, plants, threshold)
        if outcome['success_rate'] >= target:
            return float(threshold), outcome, True
    return float(grid[-1]), bw.outcomes(probability, success, plants, grid[-1]), False


def conditioned_fixed(success, plants, target, per_history=False, grid=br.GRID):
    """b3_waiting_baselines.conditioned_fixed with the target as a parameter."""
    sub = success[:, plants, :]
    rates = sub.mean(1)
    completion = np.where(sub, (grid + br.WINDOW_MIN)[None, None, :], br.DEADLINE_MIN).mean(1)
    pooled_s = (rates[0][:, None] + rates[1][None, :]) / 2
    pooled_c = (completion[0][:, None] + completion[1][None, :]) / 2
    feasible = ((rates[0][:, None] >= target) & (rates[1][None, :] >= target)) if per_history else pooled_s >= target
    if feasible.any():
        i, j = min(np.argwhere(feasible).tolist(), key=lambda ij: (pooled_c[ij[0], ij[1]], grid[ij[0]] + grid[ij[1]], grid[ij[0]]))
        reached = True
    else:
        i = j = len(grid) - 1
        reached = False
    return dict(waits_min=[float(grid[i]), float(grid[j])], target_reached=reached,
                calibration=dict(success_rate=float(pooled_s[i, j]), mean_completion_min=float(pooled_c[i, j])))


def choose(probability, success, cal, grid, target):
    """Calibration choices of every rule for one threshold grid and calibration target."""
    choices = {}
    for name in ('current', 'history'):
        threshold, outcome, reached = first_feasible(probability[name], success, cal, grid, target)
        choices[name] = dict(threshold=threshold, target_reached=reached, calibration=outcome)
    scales = []
    for tau in bw.SMOOTH_TAUS:
        threshold, outcome, reached = first_feasible(probability[f'smoothed_{tau:g}'], success, cal, grid, target)
        scales.append(dict(tau_min=tau, threshold=threshold, target_reached=reached, calibration=outcome))
    reaching = [s for s in scales if s['target_reached']]
    choices['smoothed'] = (min(reaching, key=lambda s: (s['calibration']['mean_completion_min'], s['tau_min'])) if reaching
                           else max(scales, key=lambda s: s['calibration']['success_rate']))
    rates = success[:, cal, :].mean(axis=(0, 1))
    hits = np.flatnonzero(rates >= target)
    choices['fixed'] = dict(wait_min=float(br.GRID[hits[0]] if hits.size else br.GRID[-1]), target_reached=bool(hits.size))
    choices['conditioned'] = conditioned_fixed(success, cal, target)
    choices['conditioned_per_history'] = conditioned_fixed(success, cal, target, per_history=True)
    return choices


def evaluate(probability, success, ev, choices):
    """Per-plant (ok, completion) arrays of every rule on the evaluation samples."""
    out = {name: rb.plant_outcomes(probability[name], success, ev, choices[name]['threshold']) for name in ('current', 'history')}
    out['smoothed'] = rb.plant_outcomes(probability[f"smoothed_{choices['smoothed']['tau_min']:g}"], success, ev,
                                        choices['smoothed']['threshold'])
    out['fixed'] = rb.fixed_plant_outcomes(success, ev, choices['fixed']['wait_min'])
    for name in ('conditioned', 'conditioned_per_history'):
        out[name] = bl.plant_fixed(success, ev, choices[name]['waits_min'])
    return out


def paired(outcomes):
    per_sample = {name: (ok.mean(0), completion.mean(0)) for name, (ok, completion) in outcomes.items()}
    return {f'{a}-{b}': dict(success=per_sample[a][0] - per_sample[b][0], completion=per_sample[a][1] - per_sample[b][1])
            for a, b in PAIRS}


def run_repeat(r, noise):
    tables = dict(np.load(br.TABLES))
    success = br.success_table(tables)
    split_seed, noise_seed = rb.seeds(r)
    cal, ev = bw.split_samples(success.shape[1], seed=split_seed)
    observed = bw.noisy_observations(tables, noise, seed=noise_seed)
    on_cal = bw.estimator_probabilities(tables, success, noise, cal, cal, observed)
    on_ev = bw.estimator_probabilities(tables, success, noise, cal, ev, observed)
    oracle = rb.summarise(*rb.plant_outcomes(success[:, ev, :].astype(float), success, ev, 1.))
    cells = {}
    for grid_name, grid in GRIDS.items():
        for target in TARGETS:
            choices = json.loads(json.dumps(choose(on_cal, success, cal, grid, target)))
            outcomes = evaluate(on_ev, success, ev, choices)
            differences = paired(outcomes)
            cell = dict(choices=choices, rules={name: rb.summarise(*pair) for name, pair in outcomes.items()},
                        paired={pair: {key: float(v.mean()) for key, v in d.items()} for pair, d in differences.items()})
            cell['rules']['oracle'] = oracle
            if r == 0:
                cell['bootstrap'] = rb.bootstrap(differences)
            cells[f'{grid_name}/{target:.2f}'] = cell
    return dict(r=r, noise=noise, split_seed=split_seed, noise_seed=noise_seed, cells=cells)


def _job(job):
    return run_repeat(*job)


def summarize(repeats):
    out = {}
    for noise in (f'{n:g}' for n in NOISES):
        out[noise] = {}
        for key in repeats['0'][noise]['cells']:
            rows = [repeats[str(r)][noise]['cells'][key] for r in range(rb.REPEATS)]
            block = dict(
                reached_on_evaluation={name: int(sum(row['rules'][name]['all']['success_rate'] >= br.Q for row in rows))
                                       for name in RULES},
                target_reached_on_calibration={name: int(sum(row['choices'][name]['target_reached'] for row in rows))
                                               for name in RULES},
                success={name: rb.stats([row['rules'][name]['all']['success_rate'] for row in rows]) for name in RULES},
                completion={name: rb.stats([row['rules'][name]['all']['mean_completion_min'] for row in rows]) for name in RULES},
                paired={pair: {k: rb.stats([row['paired'][pair][k] for row in rows]) for k in ('success', 'completion')}
                        for pair in rows[0]['paired']},
                faster={pair: int(sum(row['paired'][pair]['completion'] < 0 for row in rows)) for pair in rows[0]['paired']})
            out[noise][key] = block
    return out


def frontier_points():
    """Descriptive: evaluation frontier points of the frozen baselines study attaining each success level."""
    fine = json.loads(bl.RESULTS.read_text(encoding='utf-8'))['fine_grid']
    out = {}
    for noise, block in fine.items():
        out[noise] = {}
        for level in bl.LEVELS:
            row = {}
            for name, points in block['frontier'].items():
                feasible = [p for p in points if p['success_rate'] >= level]
                row[name] = min(feasible, key=lambda p: (p['mean_completion_min'], p['threshold'])) if feasible else None
            row['history_minus_smoothed'] = (row['history']['mean_completion_min'] - row['smoothed']['mean_completion_min']
                                             if row['history'] and row['smoothed'] else None)
            out[noise][f'{level:g}'] = row
    return out


def main():
    load_protocol()
    jobs = [(r, noise) for r in range(rb.REPEATS) for noise in NOISES]
    with ProcessPoolExecutor(max_workers=min(32, os.cpu_count() or 1)) as pool:
        results = list(pool.map(_job, jobs))
    repeats = {}
    for (r, noise), result in zip(jobs, results):
        repeats.setdefault(str(r), {})[f'{noise:g}'] = result
    result = dict(status='complete', protocol_sha256=protocol_sha256(),
                  frozen_utc=json.loads(FREEZE.read_text(encoding='utf-8'))['frozen_utc'],
                  repeats=repeats, summary=summarize(repeats), frontier_points=frontier_points(),
                  limitations=['Simulation of one published model structure; not evidence about measured cells.',
                               'Repeats redraw the split and observation noise only; one finite posterior set is reused.',
                               'A shared calibration margin is a sensitivity analysis, not a coverage guarantee.'])
    RESULTS.write_text(json.dumps(result, indent=2) + '\n', newline='\n')
    for noise, block in result['summary'].items():
        for key, cell in block.items():
            r0 = repeats['0'][noise]['cells'][key]
            print(noise, key, 'reached', cell['reached_on_evaluation'],
                  'H-CF', round(cell['paired']['history-conditioned']['completion']['median'], 2),
                  'S-CF', round(cell['paired']['smoothed-conditioned']['completion']['median'], 2),
                  'H-S', round(cell['paired']['history-smoothed']['completion']['median'], 2),
                  'r0', {n: (round(100 * v['all']['success_rate'], 1), round(v['all']['mean_completion_min'], 1))
                         for n, v in r0['rules'].items() if n in RULES})


if __name__ == '__main__':
    main()
