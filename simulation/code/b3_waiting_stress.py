"""Low-noise degeneracy and model mismatch for the calibrated waiting rules on B3.

Implements configs/b3_waiting_stress_protocol.json (reviewer items R6 and
R7), frozen before its only run: a noise grid with a pre-specified
bandwidth variant of the history estimator and its effective sample size,
and the frozen version-3 rules under AR(1) observation noise and a reporter
offset. Simulation of one published model structure.
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os

import numpy as np

import b3_readiness as br
import b3_waiting as bw

PROTOCOL = br.ROOT / 'configs/b3_waiting_stress_protocol.json'
FREEZE = br.OUT / 'b3_waiting_stress_protocol_freeze.json'
RESULTS = br.OUT / 'waiting_stress_results.json'
NOISE_GRID = (0., .001, .0025, .005, .01)
BANDWIDTH_FLOOR = .005
RHO, AR_SEED, OFFSET = .8, 20261009, .0025
ESS_MIN = 90.


def protocol_sha256():
    return hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()


def load_protocol():
    if protocol_sha256() != json.loads(FREEZE.read_text(encoding='utf-8'))['protocol_sha256']:
        raise SystemExit('Protocol differs from its freeze record; refusing to run.')
    bw.load_protocol()
    return json.loads(PROTOCOL.read_text(encoding='utf-8'))


def ar1_observations(tables, sd=.005):
    fret = tables['history_fret'].astype(float)
    noise = np.empty_like(fret)
    scale = np.sqrt(1 - RHO ** 2) * sd
    for h in range(fret.shape[0]):
        for plant in range(fret.shape[1]):
            eps = np.random.default_rng(AR_SEED + 2 * plant + h).normal(0., 1., fret.shape[2])
            noise[h, plant, 0] = sd * eps[0]
            for k in range(1, fret.shape[2]):
                noise[h, plant, k] = RHO * noise[h, plant, k - 1] + scale * eps[k]
    return fret + noise


def offset_observations(tables):
    return bw.noisy_observations(tables, .005) + OFFSET


def history_probabilities(tables, success, observed, sd, reference, plants):
    fret = tables['history_fret'].astype(float)
    out = np.zeros((fret.shape[0], len(plants), fret.shape[2]))
    for h in range(fret.shape[0]):
        for i, p in enumerate(plants):
            keep = reference[reference != p]
            out[h, i] = bw.reference_history_probability(observed[h, p], fret[h, keep], success[h, keep], sd)
    return out


def median_ess(tables, observed, sd, reference, plants):
    fret, k = tables['history_fret'][1].astype(float), int(np.flatnonzero(br.GRID == ESS_MIN)[0])
    values = []
    for p in plants:
        log_w = -np.cumsum((fret[reference] - observed[1, p]) ** 2, axis=1)[:, k] / (2 * sd ** 2)
        w = np.exp(log_w - log_w.max())
        values.append(w.sum() ** 2 / (w ** 2).sum())
    return float(np.median(values))


def noise_cell(noise):
    tables = dict(np.load(br.TABLES))
    success = br.success_table(tables)
    cal, ev = bw.split_samples(success.shape[1])
    choices = bw.calibration_choices(tables, noise)
    evaluation = bw.evaluate(tables, noise, choices)
    observed = bw.noisy_observations(tables, noise)
    variant_sd = max(noise, BANDWIDTH_FLOOR)
    threshold, calibration, reached = bw.calibrate_threshold(
        history_probabilities(tables, success, observed, variant_sd, cal, cal), success, cal)
    variant = dict(likelihood_sd=variant_sd, threshold=threshold, target_reached=reached, calibration=calibration,
                   evaluation=bw.outcomes(history_probabilities(tables, success, observed, variant_sd, cal, ev), success, ev, threshold))
    ess = dict(default=median_ess(tables, observed, bw.likelihood_sd(noise), cal, ev),
               variant=median_ess(tables, observed, variant_sd, cal, ev))
    return dict(noise=noise, choices=choices, evaluation=evaluation, history_bandwidth_variant=variant, median_ess_90min=ess)


def mismatch():
    tables = dict(np.load(br.TABLES))
    v3 = json.loads(bw.RESULTS.read_text(encoding='utf-8'))['noise']['0.005']['choices']
    return {name: bw.evaluate(tables, .005, v3, observed=observed)
            for name, observed in (('ar1', ar1_observations(tables)), ('offset', offset_observations(tables)))}


def main():
    load_protocol()
    with ProcessPoolExecutor(max_workers=min(len(NOISE_GRID), os.cpu_count() or 1)) as pool:
        cells = list(pool.map(noise_cell, NOISE_GRID))
    result = dict(status='complete', protocol_sha256=protocol_sha256(),
                  frozen_utc=json.loads(FREEZE.read_text(encoding='utf-8'))['frozen_utc'],
                  noise_grid={f'{noise:g}': cell for noise, cell in zip(NOISE_GRID, cells)}, mismatch=mismatch(),
                  limitations=['Simulation of one published model structure; not evidence about measured cells.',
                               'Mismatch changes only the observation model; dynamics and task are unchanged.'])
    RESULTS.write_text(json.dumps(result, indent=2) + '\n', newline='\n')
    for noise, cell in result['noise_grid'].items():
        e, v = cell['evaluation'], cell['history_bandwidth_variant']
        print(noise, {r: (round(100 * e[r]['success_rate'], 1), round(e[r]['mean_completion_min'], 1))
                      for r in ('current', 'smoothed', 'history', 'fixed_calibrated')},
              'reached', {r: cell['choices'][r]['target_reached'] for r in ('current', 'smoothed', 'history')},
              'variant', (v['threshold'], v['target_reached'], round(100 * v['evaluation']['success_rate'], 1),
                          round(v['evaluation']['mean_completion_min'], 1)), 'ess', {k: round(x, 1) for k, x in cell['median_ess_90min'].items()})
    for name, e in result['mismatch'].items():
        print(name, {r: (round(100 * e[r]['success_rate'], 1), round(e[r]['mean_completion_min'], 1))
                     for r in ('current', 'smoothed', 'history', 'fixed_calibrated', 'oracle')})


if __name__ == '__main__':
    main()
