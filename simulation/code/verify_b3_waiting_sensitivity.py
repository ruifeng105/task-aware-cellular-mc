"""Audit the waiting sensitivity study without changing any setting.

Checks that the kappa helper implements rise >= kappa x naive rise exactly,
that kappa 0.5 at noise 0.005 reproduces version 3, that calibration ignores
evaluation plants at a non-default setting, and recomputes the evaluation of
another setting from its saved choices.
"""
import json
import sys

import numpy as np

import b3_readiness as br
import b3_waiting as bw
import b3_waiting_sensitivity as sv


def kappa_helper_check(tables):
    for kappa in (.4, .6):
        direct = tables['rise'] >= kappa * tables['naive_rise'][None, :, None]
        assert np.array_equal(br.success_table(sv.with_kappa(tables, kappa)), direct), kappa
    assert np.array_equal(br.success_table(sv.with_kappa(tables, .5)), br.success_table(tables))
    return dict(kappas_checked=[.4, .5, .6])


def version3_check(saved):
    v3 = json.loads(bw.RESULTS.read_text(encoding='utf-8'))['noise']['0.005']
    cell = saved['cells']['0.5']['0.005']
    assert cell['choices'] == v3['choices']
    names = {'current': 'current', 'smoothed': 'smoothed', 'history': 'history', 'fixed': 'fixed_calibrated', 'oracle': 'oracle'}
    worst = max(abs(cell['rules'][ours]['all'][key] - v3['evaluation'][theirs][key])
                for ours, theirs in names.items() for key in ('success_rate', 'mean_completion_min'))
    assert worst < 1e-12, worst
    return dict(max_abs_difference=worst)


def leakage_check(tables, saved, kappa=.4, noise=.01):
    scaled = sv.with_kappa(tables, kappa)
    split = bw.split_samples(scaled['naive_rise'].size)
    flipped = dict(scaled)
    flipped['rise'] = scaled['rise'].copy()
    flipped['rise'][:, split[1], :] = np.where(flipped['rise'][:, split[1], :] > 0, -1., 1.)
    choices = bw.calibration_choices(flipped, noise, split, bw.noisy_observations(tables, noise))
    assert choices == saved['cells'][f'{kappa:g}'][f'{noise:g}']['choices']
    return dict(kappa=kappa, noise=noise, choices_unchanged=True)


def recompute_check(tables, saved, kappa=.6, noise=.01):
    cell = saved['cells'][f'{kappa:g}'][f'{noise:g}']
    fresh = sv.evaluate_cell(tables, kappa, noise, cell['choices'])
    worst = 0.
    for rule, parts in cell['rules'].items():
        for part, row in parts.items():
            for key, value in row.items():
                worst = max(worst, abs(fresh['rules'][rule][part][key] - value))
    assert worst < 1e-12, worst
    return dict(kappa=kappa, noise=noise, max_abs_difference=worst)


def main():
    sv.load_protocol()
    tables = dict(np.load(br.TABLES))
    saved = json.loads(sv.RESULTS.read_text(encoding='utf-8'))
    report = dict(status='passed', protocol_sha256=sv.protocol_sha256(), kappa_helper=kappa_helper_check(tables),
                  version3=version3_check(saved), leakage=leakage_check(tables, saved),
                  recomputation=recompute_check(tables, saved),
                  scope='Simulation of the published B3 model; not evidence about measured cells.')
    (br.OUT / 'waiting_sensitivity_verification_results.json').write_text(json.dumps(report, indent=2) + '\n', newline='\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    sys.exit(main())
