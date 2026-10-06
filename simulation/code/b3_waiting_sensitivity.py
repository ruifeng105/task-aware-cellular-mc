"""Sensitivity of the calibrated waiting rules on B3 to the task threshold and observation noise.

Implements configs/b3_waiting_sensitivity_protocol.json, frozen before its
only run: kappa in {0.4, 0.5, 0.6} x observation noise in {0.005, 0.01} on
the primary calibration/evaluation split with the version-3 rules. kappa =
0.5 at noise 0.005 reproduces version 3. Simulation of one published model
structure, not evidence about measured cells.
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os

import numpy as np

import b3_readiness as br
import b3_waiting as bw
import b3_waiting_robustness as rb

PROTOCOL = br.ROOT / 'configs/b3_waiting_sensitivity_protocol.json'
FREEZE = br.OUT / 'b3_waiting_sensitivity_protocol_freeze.json'
RESULTS = br.OUT / 'waiting_sensitivity_results.json'
KAPPAS = (.4, .5, .6)
NOISES = (.005, .01)
PAIRS = (('history', 'smoothed'), ('history', 'fixed'))


def protocol_sha256():
    return hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()


def load_protocol():
    if protocol_sha256() != json.loads(FREEZE.read_text(encoding='utf-8'))['protocol_sha256']:
        raise SystemExit('Protocol differs from its freeze record; refusing to run.')
    bw.load_protocol()
    return json.loads(PROTOCOL.read_text(encoding='utf-8'))


def with_kappa(tables, kappa):
    """Copy of the tables whose success table is rise >= kappa x naive rise (version 2 uses kappa = 0.5)."""
    return dict(tables, naive_rise=tables['naive_rise'] * (kappa / br.KAPPA))


def evaluate_cell(tables, kappa, noise, choices):
    scaled = with_kappa(tables, kappa)
    success = br.success_table(scaled)
    cal, ev = bw.split_samples(success.shape[1])
    observed = bw.noisy_observations(tables, noise)
    probability = bw.estimator_probabilities(scaled, success, noise, cal, ev, observed)
    outcomes = {name: rb.plant_outcomes(probability[name], success, ev, choices[name]['threshold']) for name in ('current', 'history')}
    outcomes['smoothed'] = rb.plant_outcomes(probability[f"smoothed_{choices['smoothed']['tau_min']:g}"], success, ev,
                                             choices['smoothed']['threshold'])
    outcomes['fixed'] = rb.fixed_plant_outcomes(success, ev, choices['fixed']['wait_min'])
    rules = {name: rb.summarise(*pair) for name, pair in outcomes.items()}
    rules['oracle'] = rb.summarise(*rb.plant_outcomes(success[:, ev, :].astype(float), success, ev, 1.))
    per_sample = {name: (ok.mean(0), completion.mean(0)) for name, (ok, completion) in outcomes.items()}
    differences = {f'{a}-{b}': dict(success=per_sample[a][0] - per_sample[b][0], completion=per_sample[a][1] - per_sample[b][1])
                   for a, b in PAIRS}
    return dict(rules=rules, paired={pair: {key: float(v.mean()) for key, v in d.items()} for pair, d in differences.items()},
                bootstrap=rb.bootstrap(differences))


def run_cell(job):
    kappa, noise = job
    tables = dict(np.load(br.TABLES))
    scaled = with_kappa(tables, kappa)
    choices = bw.calibration_choices(scaled, noise, None, bw.noisy_observations(tables, noise))
    cell = evaluate_cell(tables, kappa, noise, choices)
    reached = {name: bool(choice['target_reached']) for name, choice in choices.items()}
    return dict(kappa=kappa, noise=noise, choices=choices, reached=reached, **cell)


def main():
    load_protocol()
    jobs = [(kappa, noise) for kappa in KAPPAS for noise in NOISES]
    with ProcessPoolExecutor(max_workers=min(len(jobs), os.cpu_count() or 1)) as pool:
        results = list(pool.map(run_cell, jobs))
    cells = {}
    for (kappa, noise), result in zip(jobs, results):
        cells.setdefault(f'{kappa:g}', {})[f'{noise:g}'] = result
    result = dict(status='complete', protocol_sha256=protocol_sha256(),
                  frozen_utc=json.loads(FREEZE.read_text(encoding='utf-8'))['frozen_utc'], cells=cells,
                  limitations=['Simulation of one published model structure; not evidence about measured cells.',
                               'Primary split only; split variability at kappa 0.5, noise 0.005 is in waiting_robustness_results.json.'])
    RESULTS.write_text(json.dumps(result, indent=2) + '\n', newline='\n')
    for kappa, block in cells.items():
        for noise, cell in block.items():
            rules = cell['rules']
            print(kappa, noise, {k: (round(100 * v['all']['success_rate'], 1), round(v['all']['mean_completion_min'], 1))
                                 for k, v in rules.items()}, 'reached', cell['reached'],
                  'fixed', cell['choices']['fixed']['wait_min'], 'tau', cell['choices']['smoothed']['tau_min'])


if __name__ == '__main__':
    main()
