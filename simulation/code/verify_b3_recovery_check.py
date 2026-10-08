"""Audit the posterior predictive check of B3 recovery (configs/b3_recovery_check_protocol.json).

Checks: pinned inputs; the 25 ng/ml ratios equal the frozen natural-probe results; unit tests of the residual,
effective-sample-size and rise helpers; a fresh recomputation equals the saved results; the reading rules R1-R4
re-derived from the saved numbers with independent code.
"""
import json
import sys

import numpy as np

import b3_recovery_check as rc
import natural_probe as npb


def close(a, b, path=''):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            close(a[k], b[k], f'{path}/{k}')
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            close(x, y, f'{path}/{i}')
    elif isinstance(a, float) or isinstance(b, float):
        assert abs(a - b) <= 1e-9 + 1e-5 * abs(b), (path, a, b)
    else:
        assert a == b, (path, a, b)


def natural_probe_check(saved):
    """Measured ratios and the 3/20 B3 ratio equal the frozen natural probe. For the mixed probe the natural probe
    divides each sample's probe rise by the median naive rise over samples, whereas this protocol divides by the
    sample's own naive rise; the natural-probe definition is recomputed from the same simulation and compared."""
    probe = json.loads(npb.RESULTS.read_text(encoding='utf-8'))['probes']
    mixed, three = probe['fgf_mixed_25ng/pulse_3']['rise_over_naive'], probe['fgf_3_20_25ng/pulse_2']['rise_over_naive']
    close(saved['mixed']['25ng']['measured'], mixed['measured_median'])
    close(saved['three_twenty']['25ng']['measured'], three['measured_median'])
    close(saved['three_twenty']['25ng']['b3']['median'], three['b3_median'])
    data, _ = rc.load()
    m, p = data['fgf_mixed_25ng'], data['fgf_sp_5_25ng']
    probe_f = npb.rise(m['t'], m['f'], rc.MIXED_ONSET, m['start'])
    naive_f = npb.rise(p['t'], p['f'], 0., p['start'])
    natural_definition = float(np.median(probe_f / np.median(naive_f)))
    close(natural_definition, mixed['b3_median'])
    return dict(measured_ratios_equal=True, three_twenty_b3_equal=True,
                mixed_b3_natural_probe_definition=natural_definition,
                mixed_b3_per_sample_ratio_median=saved['mixed']['25ng']['b3']['median'])


def unit_checks():
    rng = np.random.default_rng(1)
    f = 1. + 0.1 * rng.random((5, 30))
    y = np.vstack([f[2] + 0.03, 1. + 0.5 * (f[3] - 1.) + 0.02])
    sums = rc.residual_sums(y, f)
    assert sums['offset'][0, 2] < 1e-12 and sums['raw'][0, 2] > 1e-3
    assert sums['affine'][1, 3] < 1e-12 and sums['offset'][1, 3] > 1e-6
    assert np.allclose(rc.ess(np.zeros((2, 7)), 0.01), 7.)
    t = np.arange(0., 40., 2.)
    step = np.where(t > 10., 1.2, 1.)
    assert abs(npb.rise(t, step[None], 10., 1.)[0] - 0.2) < 1e-12
    return dict(residuals=True, ess=True, rise=True)


def reading_check(saved):
    r = saved['reading']
    for label in ('mixed', 'three_twenty'):
        for d in rc.DOSES:
            b = saved[label][d]
            out = b['interval95'][1] < b['b3']['q05'] or b['interval95'][0] > b['b3']['q95']
            assert r['R1_inconsistent'][f'{label}/{d}'] == out, (label, d)
            inside = b['null']['q25'] <= b['measured'] <= b['null']['q75']
            assert r['R3_indistinguishable_from_no_response'][f'{label}/{d}'] == inside, (label, d)
    i = r['R1_inconsistent']
    assert r['R2_localized_to_long_strong_commands'] == (i['mixed/25ng'] and i['mixed/250ng'] and not i['mixed/2-5ng']
                                                         and not i['three_twenty/2-5ng'] and not i['three_twenty/250ng'])
    fitted = ['fgf_3_20_2-5ng', 'fgf_3_20_250ng', 'fgf_sus_2-5ng', 'fgf_sus_250ng']
    assert sorted(r['fitted_conditions']) == sorted(fitted)
    c = saved['collapse']
    assert r['R4_collapse_from_unrepresented_cell_deviations'] == all(
        c[k]['raw']['ess'] <= 3 and c[k]['affine']['best_residual_over_noise'] > 2 and c[k]['posterior_spread'] < c[k]['noise_floor']
        for k in fitted)
    return dict(rules_rederived=['R1', 'R2', 'R3', 'R4'])


def main():
    rc.load_protocol()
    saved = json.loads(rc.RESULTS.read_text(encoding='utf-8'))
    fresh = rc.compute()
    close(fresh, {k: saved[k] for k in fresh})
    report = dict(status='passed', protocol_sha256=rc.protocol_sha256(), natural_probe=natural_probe_check(saved),
                  units=unit_checks(), recompute=dict(identical_within='1e-9 + 1e-5 |value|'), reading=reading_check(saved),
                  scope='Descriptive check of the published B3 posterior against measured cells; nothing is re-fitted.')
    (rc.OUT / 'recovery_check_verification_results.json').write_text(json.dumps(report, indent=2) + '\n', newline='\n')
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
