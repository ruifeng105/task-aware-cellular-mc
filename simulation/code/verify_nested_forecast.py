"""Audit the nested P/C/O/M/H analysis without changing any setting.

Checks the phase rule at switching times, model nesting, that H reproduces M3
of the B3 reference comparison, recomputes every main-role score from the saved
models, re-reads the authors' B3 data roles, and runs a leakage test: shifting
the held-out targets of a LOPO fold must not change its selected sigma, alphas
or held-out predictions.
"""
import json
import sys

import numpy as np

import b3_forecast as bf
import expanded_fgf2 as e
import nested_forecast as nf
from history_baselines import predict, run_metric


def phase_checks():
    cases = [((10., 10., 'fgf_sp_10'), 'early_washout'), ((40., 10., 'fgf_sp_10'), 'early_washout'),
             ((42., 10., 'fgf_sp_10'), 'late'), ((30., 10., 'fgf_sp_60'), 'stimulation'),
             ((60., 10., 'fgf_sp_60'), 'early_washout'), ((14., 10., 'fgf_mixed'), 'next_command'),
             ((12., 10., 'fgf_mixed'), 'early_washout'), ((22., 2., 'fgf_3_20'), 'next_command'),
             ((20., 2., 'fgf_3_20'), 'early_washout'), ((24., 2., 'fgf_3_20'), 'stimulation'),
             ((100., 10., 'fgf_sus'), 'stimulation')]
    for args, expected in cases:
        assert nf.phase(*args) == expected, (args, nf.phase(*args), expected)
    return dict(cases=len(cases))


def nesting_checks():
    for h in (2., 10.):
        f = nf.nested_features(h)
        assert set(f['C']) < set(f['O']) < set(f['M']) < set(f['H'])
        assert set(f['H']) == set(e.feature_sets(h)['causal_history'] + ['b3_delta'])
    return dict(nested=True, H_equals_M3_features=True)


def leakage_check(rows10, groups, protocol, held='fgf_sp_5'):
    base = nf.lopo_fold(rows10, groups, held, protocol)
    shifted = rows10.copy()
    shifted.loc[shifted.run_id.isin(groups[held]), 'target'] += 1.
    moved = nf.lopo_fold(shifted, groups, held, protocol)
    assert base['sigma'] == moved['sigma'] and base['alpha'] == moved['alpha']
    for name in base['prediction']:
        np.testing.assert_allclose(base['prediction'][name], moved['prediction'][name], atol=1e-12)
    return dict(held_out=held, selection_unchanged=True, predictions_unchanged=True)


def phase_detail_check(saved):
    """Per-condition phase errors pool (condition-equal) back to the frozen phase scores."""
    import nested_phase_detail as pd_
    detail = json.loads(pd_.DETAIL.read_text(encoding='utf-8'))
    worst = 0.
    for phase, conditions in detail['phases'].items():
        frozen = saved['phases']['10'][phase]
        assert len(conditions) == frozen['n_conditions'] and sum(c['rows'] for c in conditions.values()) == frozen['n_rows']
        for name, value in frozen['rmse'].items():
            pooled = float(np.sqrt(np.mean([c['rmse'][name] ** 2 for c in conditions.values()])))
            worst = max(worst, abs(pooled - value))
    assert worst < 1e-12, worst
    assert sorted(detail['phases']['next_command']) == ['fgf_mixed_0-25ng', 'fgf_mixed_25ng']
    return dict(max_abs_pooling_difference=worst)


def main():
    saved = json.loads(nf.RESULTS.read_text(encoding='utf-8'))
    forecast = json.loads(bf.RESULTS.read_text(encoding='utf-8'))
    protocol, roles, blocks, population = nf.load_inputs()
    worst, m3 = 0., 0.
    for h, analysis in saved['main'].items():
        rows = bf.forecast_rows(blocks, population, float(h), [nf.sigma_value(analysis['sigma'])])
        data = rows.assign(b3_delta=rows[f'b3_delta_{analysis["sigma"]}'])
        for role, models in analysis['scores'].items():
            part = data[data.run_id.isin(roles[role])]
            for name, metric in models.items():
                yp = part.current_y.to_numpy() if name == 'P' else predict(part, analysis['models'][name])
                worst = max(worst, abs(run_metric(part, yp)['run_equal_mse'] - metric['run_equal_mse']))
            m3 = max(m3, abs(models['H']['run_equal_mse'] - forecast['scores'][h][role]['M3_history_plus_b3']['run_equal_mse']))
    assert worst < 1e-12 and m3 < 1e-10, (worst, m3)
    rows10 = bf.forecast_rows(blocks, population, 10., protocol['selection']['sigma_grid'])
    leakage = leakage_check(rows10, nf.lopo_groups(blocks, roles), protocol)
    roles_b3 = nf.b3_data_roles()
    assert roles_b3 == saved['b3_roles'] and roles_b3['fitted'] == ['sus_2-5ng', 'sus_250ng', '3_20_2-5ng', '3_20_250ng']
    report = dict(status='passed', protocol_sha256=nf.protocol_sha256(), phases=phase_checks(), nesting=nesting_checks(),
                  scores_recomputed=sum(len(m) for a in saved['main'].values() for m in a['scores'].values()),
                  max_abs_score_difference=worst, max_abs_H_minus_M3=m3, lopo_leakage=leakage, b3_roles=roles_b3,
                  phase_detail=phase_detail_check(saved))
    (nf.OUT / 'nested_verification_results.json').write_text(json.dumps(report, indent=2) + '\n', newline='\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    sys.exit(main())
