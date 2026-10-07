"""Audit the longer-horizon, noise-floor, crossing, bootstrap and phase-gated forecast study.

Checks that horizons 2 and 10 and the O/M LOPO folds reproduce the frozen
nested results, tests the noise-floor estimator and the phase interaction
features on synthetic data, recomputes skill and reduction arithmetic from
the saved per-condition errors, and checks that persistence never anticipates
a crossing (balanced accuracy 0.5).
"""
import json
import sys

import numpy as np
import pandas as pd

import nested_extensions as nx

VERIFY = nx.nf.OUT / 'nested_extensions_verification_results.json'


def reproduction_check(saved):
    worst = max(saved['horizons'][h]['reproduction_max_abs_rmse_difference'] for h in ('2', '10'))
    worst_lopo = max(f['reproduction_max_abs_difference'] for f in saved['lopo'].values())
    assert worst < 1e-10 and worst_lopo < 1e-10, (worst, worst_lopo)
    return dict(horizons_2_10_max_abs_rmse_difference=worst, lopo_O_M_max_abs_difference=worst_lopo)


def noise_floor_check():
    rng = np.random.default_rng(5)
    t = np.arange(0., 200., 2.)
    y = 1 + .05 * np.exp(-((t - 60) / 40) ** 2) + rng.normal(0, .01, (400, t.size))
    estimate = nx.noise_floor({'synthetic': dict(y=y)})['synthetic']
    assert abs(estimate / .01 - 1) < .05, estimate
    return dict(true_sd=.01, estimate=estimate)


def phase_feature_check():
    rows = pd.DataFrame(dict(time_min=[10., 20., 40., 112.], history_id=['fgf_mixed'] * 4,
                             b3_delta=[.1, .2, .3, .4], current_y=[1.1, 1.2, 1.3, 1.4], slope=[.01, .02, .03, .04]))
    out = nx.with_phases(rows, 10.)
    assert list(out.phase) == ['early_washout', 'next_command', 'stimulation', 'next_command'], list(out.phase)
    assert np.allclose(out.nc_b3_delta, [0, .2, 0, .4]) and np.allclose(out.st_current_y, [0, 0, 1.3, 0])
    return dict(phase_indicators_from_schedule=True)


def arithmetic_check(saved):
    floor, worst = saved['noise_floor'], 0.
    for h, block in saved['horizons'].items():
        for role, r in block['roles'].items():
            per_run = r['per_run_mse']
            for name, value in r['skill'].items():
                worst = max(worst, abs(value - nx.skill(per_run[name], per_run['P'], floor)))
            for pair, value in r['reduction'].items():
                a, b = pair.split('_vs_')
                worst = max(worst, abs(value - (1 - r['rmse'][a] / r['rmse'][b])))
            if 'crossing' in r and r['crossing']['P']['balanced_accuracy'] is not None:
                worst = max(worst, abs(r['crossing']['P']['balanced_accuracy'] - .5))
    assert worst < 1e-12, worst
    return dict(max_abs_difference=worst)


def main():
    nx.load_protocol()
    saved = json.loads(nx.RESULTS.read_text(encoding='utf-8'))
    checks = dict(reproduction=reproduction_check(saved), noise_floor=noise_floor_check(),
                  phases=phase_feature_check(), arithmetic=arithmetic_check(saved))
    result = dict(status='passed', protocol_sha256=nx.protocol_sha256(), checks=checks)
    VERIFY.write_text(json.dumps(result, indent=2) + '\n', newline='\n')
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
