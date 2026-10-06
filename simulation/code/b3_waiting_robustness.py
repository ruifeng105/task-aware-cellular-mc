"""Robustness of the calibrated waiting rules on B3 over repeated splits and noise draws.

Implements configs/b3_waiting_robustness_protocol.json, a supplement to
readiness protocol version 3 frozen before its only run. Repeat 0 reproduces
version 3; repeats 1-19 redraw the calibration/evaluation split and the
observation noise. Reports outcomes per history, paired per-sample
differences between rules, bootstrap intervals for repeat 0, and an
objective-aligned threshold selection as a sensitivity analysis. Stability of
one simulated model structure, not evidence about measured cells.
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os

import numpy as np

import b3_readiness as br
import b3_waiting as bw

PROTOCOL = br.ROOT / 'configs/b3_waiting_robustness_protocol.json'
FREEZE = br.OUT / 'b3_waiting_robustness_protocol_freeze.json'
RESULTS = br.OUT / 'waiting_robustness_results.json'
REPEATS, BOOTSTRAP, BOOT_SEED = 20, 2000, 20261008
PAIRS = (('history', 'smoothed'), ('history', 'current'), ('history', 'fixed'))


def protocol_sha256():
    return hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()


def load_protocol():
    if protocol_sha256() != json.loads(FREEZE.read_text(encoding='utf-8'))['protocol_sha256']:
        raise SystemExit('Protocol differs from its freeze record; refusing to run.')
    bw.load_protocol()
    return json.loads(PROTOCOL.read_text(encoding='utf-8'))


def seeds(r):
    """Split seed and observation seed of repeat r; repeat 0 keeps the version-2/3 draws (None)."""
    return (bw.SPLIT_SEED, None) if r == 0 else (bw.SPLIT_SEED + r, br.SEED + 10000 * r)


def plant_outcomes(probability, success, plants, threshold):
    k = np.apply_along_axis(br.first_crossing, -1, probability, threshold)
    ok = np.take_along_axis(success[:, plants, :], k[..., None], -1)[..., 0]
    return ok, np.where(ok, br.GRID[k] + br.WINDOW_MIN, br.DEADLINE_MIN)


def fixed_plant_outcomes(success, plants, wait):
    k = int(np.flatnonzero(br.GRID == wait)[0])
    ok = success[:, plants, k]
    return ok, np.where(ok, wait + br.WINDOW_MIN, br.DEADLINE_MIN)


def aligned_threshold(probability, success, plants):
    """Among thresholds whose success reaches the target, the one with the smallest mean completion."""
    feasible = []
    for threshold in bw.THRESHOLDS:
        outcome = bw.outcomes(probability, success, plants, threshold)
        if outcome['success_rate'] >= br.Q:
            feasible.append((outcome['mean_completion_min'], threshold, outcome))
    if not feasible:
        return bw.THRESHOLDS[-1], bw.outcomes(probability, success, plants, bw.THRESHOLDS[-1]), False
    _, threshold, outcome = min(feasible, key=lambda f: (f[0], f[1]))
    return threshold, outcome, True


def aligned_choices(tables, noise, split, observed):
    success = br.success_table(tables)
    cal = split[0]
    probability = bw.estimator_probabilities(tables, success, noise, cal, cal, observed)
    choices = {}
    for name in ('current', 'history'):
        threshold, outcome, reached = aligned_threshold(probability[name], success, cal)
        choices[name] = dict(threshold=threshold, target_reached=reached, calibration=outcome)
    pairs = []
    for tau in bw.SMOOTH_TAUS:
        threshold, outcome, reached = aligned_threshold(probability[f'smoothed_{tau:g}'], success, cal)
        pairs.append(dict(tau_min=tau, threshold=threshold, target_reached=reached, calibration=outcome))
    reaching = [p for p in pairs if p['target_reached']]
    choices['smoothed'] = (min(reaching, key=lambda p: (p['calibration']['mean_completion_min'], p['tau_min'])) if reaching
                           else max(pairs, key=lambda p: p['calibration']['success_rate']))
    rates = success[:, cal, :].mean(axis=(0, 1))
    completion = np.where(success[:, cal, :], (br.GRID + br.WINDOW_MIN)[None, None, :], br.DEADLINE_MIN).mean(axis=(0, 1))
    hits = np.flatnonzero(rates >= br.Q)
    wait = float(br.GRID[hits[np.argmin(completion[hits])]]) if hits.size else float(br.GRID[-1])
    choices['fixed'] = dict(wait_min=wait, target_reached=bool(hits.size))
    return json.loads(json.dumps(choices))


def summarise(ok, completion):
    parts = {'all': (ok, completion)}
    parts.update({name: (ok[h], completion[h]) for h, name in enumerate(br.HISTORIES)})
    return {part: dict(success_rate=float(o.mean()), mean_completion_min=float(c.mean())) for part, (o, c) in parts.items()}


def paired(outcomes):
    """Differences per evaluation parameter sample, each averaged over both histories."""
    per_sample = {name: (ok.mean(0), completion.mean(0)) for name, (ok, completion) in outcomes.items()}
    return {f'{a}-{b}': dict(success=per_sample[a][0] - per_sample[b][0], completion=per_sample[a][1] - per_sample[b][1])
            for a, b in PAIRS}


def bootstrap(differences):
    rng = np.random.default_rng(BOOT_SEED)
    n = len(next(iter(differences.values()))['completion'])
    index = rng.integers(0, n, (BOOTSTRAP, n))
    return {pair: {key: np.percentile(values[index].mean(1), [2.5, 97.5]).tolist() for key, values in d.items()}
            for pair, d in differences.items()}


def run_repeat(r, noise):
    tables = dict(np.load(br.TABLES))
    success = br.success_table(tables)
    split_seed, noise_seed = seeds(r)
    split = bw.split_samples(success.shape[1], seed=split_seed)
    observed = bw.noisy_observations(tables, noise, seed=noise_seed)
    choices = bw.calibration_choices(tables, noise, split, observed)
    aligned = aligned_choices(tables, noise, split, observed)
    cal, ev = split
    probability = bw.estimator_probabilities(tables, success, noise, cal, ev, observed)

    def evaluate(chosen):
        out = {name: plant_outcomes(probability[name], success, ev, chosen[name]['threshold']) for name in ('current', 'history')}
        out['smoothed'] = plant_outcomes(probability[f"smoothed_{chosen['smoothed']['tau_min']:g}"], success, ev,
                                         chosen['smoothed']['threshold'])
        out['fixed'] = fixed_plant_outcomes(success, ev, chosen['fixed']['wait_min'])
        return out

    heuristic, objective = evaluate(choices), evaluate(aligned)
    rules = {name: summarise(*pair) for name, pair in heuristic.items()}
    rules['oracle'] = summarise(*plant_outcomes(success[:, ev, :].astype(float), success, ev, 1.))
    rules.update({f'{name}_aligned': summarise(*pair) for name, pair in objective.items()})
    differences = paired(heuristic)
    mean = lambda d: {pair: {key: float(v.mean()) for key, v in values.items()} for pair, values in d.items()}
    result = dict(r=r, noise=noise, split_seed=split_seed, noise_seed=noise_seed, choices=choices, aligned=aligned,
                  rules=rules, paired=mean(differences), paired_aligned=mean(paired(objective)))
    if r == 0:
        result['bootstrap'] = bootstrap(differences)
    return result


def _job(job):
    return run_repeat(*job)


def stats(values):
    v = np.asarray(values, dtype=float)
    return dict(median=float(np.median(v)), min=float(v.min()), max=float(v.max()))


def summarize(repeats):
    out = {}
    for noise in (f'{n:g}' for n in bw.NOISE_LEVELS):
        rows = [repeats[str(r)][noise] for r in range(REPEATS)]
        rules = {rule: {part: {key: stats([row['rules'][rule][part][key] for row in rows]) for key in values}
                        for part, values in parts.items()} for rule, parts in rows[0]['rules'].items()}
        pairs = {kind: {pair: {key: stats([row[kind][pair][key] for row in rows]) for key in values}
                        for pair, values in rows[0][kind].items()} for kind in ('paired', 'paired_aligned')}
        out[noise] = dict(rules=rules, **pairs,
                          history_target_reached=int(sum(row['choices']['history']['target_reached'] for row in rows)),
                          history_faster_than_fixed=int(sum(row['paired']['history-fixed']['completion'] < 0 for row in rows)),
                          smoothed_scales=sorted({row['choices']['smoothed']['tau_min'] for row in rows}),
                          fixed_waits=sorted({row['choices']['fixed']['wait_min'] for row in rows}))
    return out


def main():
    load_protocol()
    jobs = [(r, noise) for r in range(REPEATS) for noise in bw.NOISE_LEVELS]
    with ProcessPoolExecutor(max_workers=min(32, os.cpu_count() or 1)) as pool:
        results = list(pool.map(_job, jobs))
    repeats = {}
    for (r, noise), result in zip(jobs, results):
        repeats.setdefault(str(r), {})[f'{noise:g}'] = result
    result = dict(status='complete', protocol_sha256=protocol_sha256(),
                  frozen_utc=json.loads(FREEZE.read_text(encoding='utf-8'))['frozen_utc'],
                  repeats=repeats, summary=summarize(repeats),
                  limitations=['Stability of one published model structure; not evidence about measured cells.',
                               'Repeats redraw the sample split and observation noise only; the simulation tables are fixed.',
                               'The 0.9 target is an empirical calibration target, not a coverage guarantee.'])
    RESULTS.write_text(json.dumps(result, indent=2) + '\n', newline='\n')
    print(json.dumps({noise: {k: v for k, v in block.items() if k in ('history_target_reached', 'history_faster_than_fixed',
                                                                      'smoothed_scales', 'fixed_waits')}
                      for noise, block in result['summary'].items()}, indent=2))


if __name__ == '__main__':
    main()
