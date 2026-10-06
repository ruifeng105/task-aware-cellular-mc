"""Audit the shared calibration margin study.

Checks that the target-parameterized conditioned fixed wait equals the frozen
one at 0.9, that the coarse grid at target 0.9 reproduces the robustness,
baselines and sensitivity results, that choices ignore evaluation labels,
and recomputes one fine-grid cell from scratch.
"""
import json
import sys

import numpy as np

import b3_readiness as br
import b3_waiting as bw
import b3_waiting_baselines as bl
import b3_waiting_margin as wm
import b3_waiting_robustness as rb
import b3_waiting_sensitivity as sv


def conditioned_check(tables):
    success = br.success_table(tables)
    cal, _ = bw.split_samples(success.shape[1])
    for per_history in (False, True):
        new = wm.conditioned_fixed(success, cal, br.Q, per_history=per_history)
        old = bl.conditioned_fixed(success, cal, per_history=per_history)
        assert new['waits_min'] == old['waits_min'] and new['target_reached'] == old['target_reached'], (new, old)
    return dict(matches_frozen_at_0_9=True)


def reproduction_check(saved):
    robust = json.loads(rb.RESULTS.read_text(encoding='utf-8'))['repeats']
    baselines = json.loads(bl.RESULTS.read_text(encoding='utf-8'))['repeats']['rows']
    sensitivity = json.loads(sv.RESULTS.read_text(encoding='utf-8'))['cells']['0.5']['0.01']['rules']
    worst = 0.
    for r in range(rb.REPEATS):
        rules = saved['repeats'][str(r)]['0.005']['cells']['coarse/0.90']['rules']
        for name in ('current', 'smoothed', 'history', 'fixed'):
            for part, row in robust[str(r)]['0.005']['rules'][name].items():
                for key, value in row.items():
                    worst = max(worst, abs(rules[name][part][key] - value))
        for part, row in baselines[r]['evaluation'].items():
            for key, value in row.items():
                worst = max(worst, abs(rules['conditioned'][part][key] - value))
    rules = saved['repeats']['0']['0.01']['cells']['coarse/0.90']['rules']
    for name in ('current', 'smoothed', 'history', 'fixed'):
        for part, row in sensitivity[name].items():
            for key, value in row.items():
                worst = max(worst, abs(rules[name][part][key] - value))
    assert worst < 1e-12, worst
    return dict(repeats=rb.REPEATS, max_abs_difference=worst)


def leakage_check(tables, saved, noise=.005, key='coarse/0.94'):
    success = br.success_table(tables)
    cal, ev = bw.split_samples(success.shape[1])
    flipped = success.copy()
    flipped[:, ev, :] = ~flipped[:, ev, :]
    probability = bw.estimator_probabilities(tables, flipped, noise, cal, cal, bw.noisy_observations(tables, noise))
    grid, target = key.split('/')
    choices = json.loads(json.dumps(wm.choose(probability, flipped, cal, wm.GRIDS[grid], float(target))))
    assert choices == saved['repeats']['0'][f'{noise:g}']['cells'][key]['choices'], key
    return dict(noise=noise, cell=key, evaluation_labels_flipped=True, choices_unchanged=True)


def recompute_check(saved, r=3, noise=.01, key='fine/0.92'):
    fresh = wm.run_repeat(r, noise)['cells'][key]
    stored = saved['repeats'][str(r)][f'{noise:g}']['cells'][key]
    worst = max(abs(fresh['rules'][name][part][k] - stored['rules'][name][part][k])
                for name in fresh['rules'] for part in fresh['rules'][name] for k in fresh['rules'][name][part])
    assert worst < 1e-12 and json.loads(json.dumps(fresh['choices'])) == stored['choices'], worst
    return dict(r=r, noise=noise, cell=key, max_abs_difference=worst)


def feasibility_check(saved):
    count = 0
    for r in range(rb.REPEATS):
        for noise in ('0.005', '0.01'):
            for key, cell in saved['repeats'][str(r)][noise]['cells'].items():
                target = float(key.split('/')[1])
                for name in ('current', 'history', 'conditioned', 'conditioned_per_history'):
                    choice = cell['choices'][name]
                    if choice['target_reached']:
                        assert choice['calibration']['success_rate'] >= target - 1e-12, (r, noise, key, name)
                        count += 1
    return dict(feasible_choices_checked=count)


def main():
    wm.load_protocol()
    tables = dict(np.load(br.TABLES))
    saved = json.loads(wm.RESULTS.read_text(encoding='utf-8'))
    report = dict(status='passed', protocol_sha256=wm.protocol_sha256(), conditioned=conditioned_check(tables),
                  reproduction=reproduction_check(saved), leakage=leakage_check(tables, saved),
                  recomputation=recompute_check(saved), feasibility=feasibility_check(saved),
                  scope='Simulation of the published B3 model; not evidence about measured cells.')
    (br.OUT / 'waiting_margin_verification_results.json').write_text(json.dumps(report, indent=2) + '\n', newline='\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    sys.exit(main())
