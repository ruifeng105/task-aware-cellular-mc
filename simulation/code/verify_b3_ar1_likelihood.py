"""Audit the covariance-aware likelihood study.

Checks the AR(1) draws against the stress study, the AR(1) log-likelihood
against an explicit multivariate Gaussian with covariance sigma^2 rho^|a-b|
and its rho = 0 limit, that the frozen arm reproduces the stress-study
mismatch, the noise estimates, that the AR(1) threshold ignores evaluation
labels, and recomputes the primary-split AR(1) arm.
"""
import json
import sys

import numpy as np

import b3_ar1_likelihood as al
import b3_readiness as br
import b3_waiting as bw
import b3_waiting_robustness as rb
import b3_waiting_stress as st


def draws_check(tables):
    assert np.array_equal(al.ar1_observations(tables, st.AR_SEED), st.ar1_observations(tables))
    return dict(r0_equals_stress_draws=True)


def likelihood_check():
    rng = np.random.default_rng(1)
    trajectories, observed = rng.normal(0., .01, (6, 9)), rng.normal(0., .01, 9)
    sigma, rho = .004, .7
    lag = np.abs(np.subtract.outer(np.arange(9), np.arange(9)))
    inverse = np.linalg.inv(sigma ** 2 * rho ** lag)
    explicit = np.array([-.5 * (observed - f) @ inverse @ (observed - f) for f in trajectories])
    recursive = al.ar1_log_weights(observed, trajectories, sigma, rho)[:, -1]
    worst = float(np.abs((explicit - explicit[0]) - (recursive - recursive[0])).max())
    assert worst < 1e-8, worst
    success = (rng.random((6, 9)) < .5).astype(float)
    white = float(np.abs(al.ar1_history_probability(observed, trajectories, success, sigma, 0.)
                         - bw.reference_history_probability(observed, trajectories, success, sigma)).max())
    assert white < 1e-12, white
    return dict(explicit_covariance_max_abs_difference=worst, rho_zero_max_abs_difference=white)


def frozen_check(saved):
    stress = json.loads(st.RESULTS.read_text(encoding='utf-8'))['mismatch']['ar1']
    rules = saved['repeats']['0']['rules']
    worst = max(abs(rules[f'frozen_{name}']['all'][k] - stress[name][k])
                for name in ('current', 'smoothed', 'history') for k in ('success_rate', 'mean_completion_min'))
    assert worst < 1e-12, worst
    return dict(max_abs_difference=worst)


def estimate_check(saved):
    worst = max(max(abs(row['estimated']['sigma'] - al.TRUE_SD) / al.TRUE_SD, abs(row['estimated']['rho'] - al.TRUE_RHO))
                for row in saved['repeats'].values())
    assert worst < .05, worst
    return dict(max_relative_sigma_or_absolute_rho_error=worst)


def leakage_check(tables, saved):
    success = br.success_table(tables)
    cal, ev = bw.split_samples(success.shape[1])
    flipped = success.copy()
    flipped[:, ev, :] = ~flipped[:, ev, :]
    observed = al.ar1_observations(tables, st.AR_SEED)
    sigma, rho = al.estimate_noise(tables, observed, cal)
    threshold, _, _ = bw.calibrate_threshold(al.ar1_probabilities(tables, flipped, observed, sigma, rho, cal, cal), flipped, cal)
    assert threshold == saved['repeats']['0']['choices']['ar1']['threshold']
    return dict(evaluation_labels_flipped=True, threshold_unchanged=True)


def recompute_check(tables, saved):
    success = br.success_table(tables)
    cal, ev = bw.split_samples(success.shape[1])
    observed = al.ar1_observations(tables, st.AR_SEED)
    choice = saved['repeats']['0']['choices']['ar1']
    fresh = rb.summarise(*rb.plant_outcomes(al.ar1_probabilities(tables, success, observed, choice['sigma'], choice['rho'], cal, ev),
                                            success, ev, choice['threshold']))
    stored = saved['repeats']['0']['rules']['ar1_history']
    worst = max(abs(fresh[part][k] - stored[part][k]) for part in fresh for k in fresh[part])
    assert worst < 1e-12, worst
    return dict(max_abs_difference=worst)


def main():
    al.load_protocol()
    tables = dict(np.load(br.TABLES))
    saved = json.loads(al.RESULTS.read_text(encoding='utf-8'))
    report = dict(status='passed', protocol_sha256=al.protocol_sha256(), draws=draws_check(tables), likelihood=likelihood_check(),
                  frozen=frozen_check(saved), estimates=estimate_check(saved), leakage=leakage_check(tables, saved),
                  recomputation=recompute_check(tables, saved),
                  scope='Simulation of the published B3 model; not evidence about measured cells.')
    (br.OUT / 'ar1_likelihood_verification_results.json').write_text(json.dumps(report, indent=2) + '\n', newline='\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    sys.exit(main())
