"""Audit the calibrated waiting study without changing any setting.

Checks that the new estimators reproduce the version-2 estimators, use only
past observations, that calibration choices ignore evaluation plants, the
fallback when no threshold reaches the target, and recomputes every
evaluation metric from the saved choices.
"""
import json
import sys

import numpy as np

import b3_readiness as br
import b3_waiting as bw


def equivalence_checks(tables, success):
    observed = br.observations(tables)
    np.testing.assert_allclose(bw.noisy_observations(tables, br.NOISE_SD), observed, atol=1e-12)
    np.testing.assert_allclose(bw.noisy_observations(tables, 0.), tables['history_fret'].astype(float), atol=1e-12)
    old = br.CurrentReadiness(tables['history_fret'], success, br.KERNEL_SD)
    new = bw.PooledReadiness(tables['history_fret'], success, br.KERNEL_SD)
    worst = 0.
    for plant in (0, 17, 503, 998):
        y = observed[1, plant]
        worst = max(worst, float(np.abs(old.probability(y, plant) - new.probability(y, exclude=plant)).max()))
        others = np.arange(success.shape[1]) != plant
        a = br.history_probability(y, tables['history_fret'][1].astype(float), success[1], plant, br.KERNEL_SD)
        b = bw.reference_history_probability(y, tables['history_fret'][1][others].astype(float), success[1][others], br.KERNEL_SD)
        worst = max(worst, float(np.abs(a - b).max()))
    assert worst < 1e-9, worst
    return dict(max_abs_difference=worst)


def causality_checks():
    series = np.linspace(1., 1.1, 20)
    changed = series.copy()
    changed[12:] += .3
    assert np.allclose(bw.ema(series, 4.)[:12], bw.ema(changed, 4.)[:12])
    rng = np.random.default_rng(3)
    traj, ok = 1 + .1 * rng.random((40, 20)), rng.random((40, 20)) > .5
    p1 = bw.reference_history_probability(series, traj, ok, .005)
    p2 = bw.reference_history_probability(changed, traj, ok, .005)
    assert np.allclose(p1[:12], p2[:12])
    return dict(ema_past_only=True, history_past_only=True)


def fallback_check():
    success = np.zeros((1, 3, 4), dtype=bool)
    probability = np.full((1, 3, 4), .2)
    threshold, outcome, reached = bw.calibrate_threshold(probability, success, np.arange(3))
    assert threshold == bw.THRESHOLDS[-1] and reached is False and outcome['success_rate'] == 0.
    return dict(fallback=True)


def leakage_check(tables, saved):
    cal, ev = bw.split_samples(tables['naive_rise'].size)
    flipped = dict(tables)
    flipped['rise'] = tables['rise'].copy()
    flipped['rise'][:, ev, :] = np.where(flipped['rise'][:, ev, :] > 0, -1., 1.)
    choices = bw.calibration_choices(flipped, br.NOISE_SD)
    assert choices == saved['noise'][f'{br.NOISE_SD:g}']['choices'], (choices, saved['noise'][f'{br.NOISE_SD:g}']['choices'])
    return dict(evaluation_labels_flipped=True, choices_unchanged=True)


def weight_degeneracy(tables, decision_min=90.):
    """Diagnostic for the noise ablation: median effective sample size of the history weights of the
    evaluation plants (30-min history) at one decision time, per noise level."""
    cal, ev = bw.split_samples(tables['naive_rise'].size)
    fret, k = tables['history_fret'][1].astype(float), int(np.flatnonzero(br.GRID == decision_min)[0])
    out = {}
    for noise in bw.NOISE_LEVELS:
        observed, sd = bw.noisy_observations(tables, noise)[1], bw.likelihood_sd(noise)
        ess = []
        for p in ev:
            log_w = -np.cumsum((fret[cal] - observed[p]) ** 2, axis=1)[:, k] / (2 * sd ** 2)
            w = np.exp(log_w - log_w.max())
            ess.append(w.sum() ** 2 / (w ** 2).sum())
        out[f'{noise:g}'] = float(np.median(ess))
    return dict(decision_min=decision_min, median_effective_sample_size=out)


def recomputation_check(tables, saved):
    worst = 0.
    for label, block in saved['noise'].items():
        evaluation = bw.evaluate(tables, float(label), block['choices'])
        for policy, row in block['evaluation'].items():
            for key in ('success_rate', 'mean_completion_min', 'mean_wait_min'):
                worst = max(worst, abs(evaluation[policy][key] - row[key]))
    assert worst < 1e-12, worst
    return dict(max_abs_difference=worst)


def main():
    br.load_protocol()
    bw.load_protocol()
    tables = dict(np.load(br.TABLES))
    success = br.success_table(tables)
    saved = json.loads(bw.RESULTS.read_text(encoding='utf-8'))
    report = dict(status='passed', protocol_sha256=bw.protocol_sha256(), equivalence=equivalence_checks(tables, success),
                  causality=causality_checks(), fallback=fallback_check(), leakage=leakage_check(tables, saved),
                  recomputation=recomputation_check(tables, saved), weight_degeneracy=weight_degeneracy(tables),
                  scope='Simulation of the published B3 model; not evidence about measured cells.')
    (br.OUT / 'waiting_verification_results.json').write_text(json.dumps(report, indent=2) + '\n', newline='\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    sys.exit(main())
