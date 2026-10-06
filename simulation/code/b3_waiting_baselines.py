"""History-conditioned fixed waits and a fine threshold grid for the calibrated waiting rules on B3.

Implements configs/b3_waiting_baselines_protocol.json (reviewer items R1 and
R8), frozen before its only run. The conditioned fixed wait uses the known
previous command, as the adaptive rules do; the fine grid checks whether the
coarse threshold grid drives differences between estimators. Simulation of
one published model structure, not evidence about measured cells.
"""
import hashlib
import json

import numpy as np

import b3_readiness as br
import b3_waiting as bw
import b3_waiting_robustness as rb
import b3_waiting_sensitivity as sv

PROTOCOL = br.ROOT / 'configs/b3_waiting_baselines_protocol.json'
FREEZE = br.OUT / 'b3_waiting_baselines_protocol_freeze.json'
RESULTS = br.OUT / 'waiting_baselines_results.json'
FINE = np.round(np.arange(.50, .995, .01), 2)
RULES = ('pooled', 'per_history')
LEVELS = (.90, .92, .94)


def protocol_sha256():
    return hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()


def load_protocol():
    if protocol_sha256() != json.loads(FREEZE.read_text(encoding='utf-8'))['protocol_sha256']:
        raise SystemExit('Protocol differs from its freeze record; refusing to run.')
    bw.load_protocol()
    return json.loads(PROTOCOL.read_text(encoding='utf-8'))


def conditioned_fixed(success, plants, per_history=False, grid=br.GRID):
    """Waits (3-min, 30-min history) minimizing the equal-weight mean completion under the success constraint."""
    sub = success[:, plants, :]
    rates = sub.mean(1)
    completion = np.where(sub, (grid + br.WINDOW_MIN)[None, None, :], br.DEADLINE_MIN).mean(1)
    pooled_s = (rates[0][:, None] + rates[1][None, :]) / 2
    pooled_c = (completion[0][:, None] + completion[1][None, :]) / 2
    feasible = ((rates[0][:, None] >= br.Q) & (rates[1][None, :] >= br.Q)) if per_history else pooled_s >= br.Q
    if feasible.any():
        i, j = min(np.argwhere(feasible).tolist(), key=lambda ij: (pooled_c[ij[0], ij[1]], grid[ij[0]] + grid[ij[1]], grid[ij[0]]))
        reached = True
    else:
        i = j = len(grid) - 1
        reached = False
    return dict(waits_min=[float(grid[i]), float(grid[j])], target_reached=reached,
                calibration=dict(success_rate=float(pooled_s[i, j]), mean_completion_min=float(pooled_c[i, j]),
                                 per_history={name: dict(success_rate=float(rates[h, (i, j)[h]]),
                                                         mean_completion_min=float(completion[h, (i, j)[h]]))
                                              for h, name in enumerate(br.HISTORIES)}))


def plant_fixed(success, plants, waits, grid=br.GRID):
    ok = np.stack([success[h, plants, int(np.flatnonzero(grid == w)[0])] for h, w in enumerate(waits)])
    return ok, np.where(ok, np.asarray(waits)[:, None] + br.WINDOW_MIN, br.DEADLINE_MIN)


def evaluate_fixed(success, plants, waits):
    return rb.summarise(*plant_fixed(success, plants, waits))


def paired_against(adaptive, fixed):
    """Per-sample (history-averaged) differences adaptive - fixed for success and completion."""
    return dict(success=adaptive[0].mean(0) - fixed[0].mean(0), completion=adaptive[1].mean(0) - fixed[1].mean(0))


def primary(tables):
    success = br.success_table(tables)
    cal, ev = bw.split_samples(success.shape[1])
    v3 = json.loads(bw.RESULTS.read_text(encoding='utf-8'))['noise']['0.005']['choices']
    probability = bw.estimator_probabilities(tables, success, .005, cal, ev)
    adaptive = {'history': rb.plant_outcomes(probability['history'], success, ev, v3['history']['threshold']),
                'smoothed': rb.plant_outcomes(probability[f"smoothed_{v3['smoothed']['tau_min']:g}"], success, ev,
                                              v3['smoothed']['threshold'])}
    conditioned = {}
    for rule in RULES:
        choice = conditioned_fixed(success, cal, per_history=rule == 'per_history')
        fixed = plant_fixed(success, ev, choice['waits_min'])
        differences = {f'{name}-conditioned': paired_against(outcome, fixed) for name, outcome in adaptive.items()}
        conditioned[rule] = dict(choice, evaluation=rb.summarise(*fixed),
                                 paired={pair: {k: float(v.mean()) for k, v in d.items()} for pair, d in differences.items()},
                                 bootstrap=rb.bootstrap(differences))
    return dict(conditioned=conditioned)


def repeats():
    robust = json.loads(rb.RESULTS.read_text(encoding='utf-8'))['repeats']
    tables = dict(np.load(br.TABLES))
    success = br.success_table(tables)
    rows = []
    for r in range(rb.REPEATS):
        cal, ev = bw.split_samples(success.shape[1], seed=rb.seeds(r)[0])
        choice = conditioned_fixed(success, cal)
        evaluation = evaluate_fixed(success, ev, choice['waits_min'])
        rules = robust[str(r)]['0.005']['rules']
        rows.append(dict(r=r, waits_min=choice['waits_min'], target_reached=choice['target_reached'], evaluation=evaluation,
                         history_minus_conditioned=rules['history']['all']['mean_completion_min'] - evaluation['all']['mean_completion_min'],
                         smoothed_minus_conditioned=rules['smoothed']['all']['mean_completion_min'] - evaluation['all']['mean_completion_min']))
    summary = {key: rb.stats([row[key] for row in rows]) for key in ('history_minus_conditioned', 'smoothed_minus_conditioned')}
    summary.update(success=rb.stats([row['evaluation']['all']['success_rate'] for row in rows]),
                   completion=rb.stats([row['evaluation']['all']['mean_completion_min'] for row in rows]),
                   reached_on_evaluation=int(sum(row['evaluation']['all']['success_rate'] >= br.Q for row in rows)),
                   history_faster=int(sum(row['history_minus_conditioned'] < 0 for row in rows)),
                   smoothed_faster=int(sum(row['smoothed_minus_conditioned'] < 0 for row in rows)))
    return dict(rows=rows, summary=summary)


def kappa_cells(tables):
    sensitivity = json.loads(sv.RESULTS.read_text(encoding='utf-8'))['cells']
    out = {}
    for kappa in sv.KAPPAS:
        success = br.success_table(sv.with_kappa(tables, kappa))
        cal, ev = bw.split_samples(success.shape[1])
        choice = conditioned_fixed(success, cal)
        evaluation = evaluate_fixed(success, ev, choice['waits_min'])
        out[f'{kappa:g}'] = dict(choice, evaluation=evaluation,
                                 history_minus_conditioned={noise: cell['rules']['history']['all']['mean_completion_min']
                                                            - evaluation['all']['mean_completion_min']
                                                            for noise, cell in sensitivity[f'{kappa:g}'].items()})
    return out


def first_feasible(probability, success, plants, grid):
    for threshold in grid:
        outcome = bw.outcomes(probability, success, plants, threshold)
        if outcome['success_rate'] >= br.Q:
            return float(threshold), outcome, True
    return float(grid[-1]), bw.outcomes(probability, success, plants, grid[-1]), False


def fine_grid(tables, noise):
    success = br.success_table(tables)
    cal, ev = bw.split_samples(success.shape[1])
    on_cal = bw.estimator_probabilities(tables, success, noise, cal, cal)
    on_ev = bw.estimator_probabilities(tables, success, noise, cal, ev)
    threshold, _, reached = first_feasible(on_cal['history'], success, cal, FINE)
    calibrated = {'history': dict(threshold=threshold, target_reached=reached,
                                  evaluation=bw.outcomes(on_ev['history'], success, ev, threshold))}
    scales = []
    for tau in bw.SMOOTH_TAUS:
        t, outcome, ok = first_feasible(on_cal[f'smoothed_{tau:g}'], success, cal, FINE)
        scales.append(dict(tau_min=tau, threshold=t, target_reached=ok, calibration=outcome))
    feasible = [s for s in scales if s['target_reached']]
    best = (min(feasible, key=lambda s: (s['calibration']['mean_completion_min'], s['tau_min'])) if feasible
            else max(scales, key=lambda s: s['calibration']['success_rate']))
    calibrated['smoothed'] = dict(tau_min=best['tau_min'], threshold=best['threshold'], target_reached=best['target_reached'],
                                  evaluation=bw.outcomes(on_ev[f"smoothed_{best['tau_min']:g}"], success, ev, best['threshold']))
    frontier = {name: [dict(threshold=float(t), **bw.outcomes(on_ev[key], success, ev, t)) for t in FINE]
                for name, key in (('history', 'history'), ('smoothed', 'smoothed_16'))}
    matched = {}
    for level in LEVELS:
        best_t = {name: min((p['mean_completion_min'] for p in points if p['success_rate'] >= level), default=None)
                  for name, points in frontier.items()}
        matched[f'{level:g}'] = dict(best_t, history_minus_smoothed=(None if None in best_t.values()
                                                                     else best_t['history'] - best_t['smoothed']))
    return dict(calibrated=calibrated, frontier=frontier, matched_success=matched)


def main():
    load_protocol()
    tables = dict(np.load(br.TABLES))
    result = dict(status='complete', protocol_sha256=protocol_sha256(),
                  frozen_utc=json.loads(FREEZE.read_text(encoding='utf-8'))['frozen_utc'],
                  primary=primary(tables), repeats=repeats(), kappa=kappa_cells(tables),
                  fine_grid={f'{noise:g}': fine_grid(tables, noise) for noise in (.005, .01)},
                  limitations=['Simulation of one published model structure; not evidence about measured cells.',
                               'The frontier uses evaluation samples descriptively; it never selects a rule.'])
    RESULTS.write_text(json.dumps(result, indent=2) + '\n', newline='\n')
    p = result['primary']['conditioned']
    print(json.dumps(dict(primary={rule: dict(waits=b['waits_min'], reached=b['target_reached'],
                                              eval={part: (round(100 * v['success_rate'], 1), round(v['mean_completion_min'], 2))
                                                    for part, v in b['evaluation'].items()},
                                              paired=b['paired'], boot=b['bootstrap']) for rule, b in p.items()},
                          repeats=result['repeats']['summary'],
                          kappa={k: dict(waits=v['waits_min'], reached=v['target_reached'],
                                         eval=(round(100 * v['evaluation']['all']['success_rate'], 1),
                                               round(v['evaluation']['all']['mean_completion_min'], 2)),
                                         hist_minus=v['history_minus_conditioned']) for k, v in result['kappa'].items()},
                          fine={n: dict(calibrated={k: (v['threshold'], round(100 * v['evaluation']['success_rate'], 1),
                                                        round(v['evaluation']['mean_completion_min'], 2)) for k, v in f['calibrated'].items()},
                                        matched=f['matched_success']) for n, f in result['fine_grid'].items()}), indent=1))


if __name__ == '__main__':
    main()
