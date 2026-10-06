"""Audit the lagged ARX baseline without changing any setting.

Checks that reporter lags only use past frames and command lags follow the
recorded schedule, model nesting, recomputes every fixed-role score from the
saved models, and runs the LOPO leakage test: shifting the held-out targets
must not change the selected window, penalty, sigma or predictions.
"""
import json
import sys

import numpy as np

import b3_forecast as bf
import calibrate_fgf2 as c
import nested_arx as ax
import nested_forecast as nf
from history_baselines import predict, run_metric


def lag_check(blocks):
    key = 'fgf_mixed_25ng'
    sub = {key: dict(blocks[key], y=blocks[key]['y'][:3])}
    rows = ax.add_lags(c.forecasting_rows(sub, 10.), sub)
    rng = np.random.default_rng(5)
    t = sub[key]['time']
    for r in rng.choice(len(rows), 25, replace=False):
        row = rows.iloc[r]
        i, cell = int(np.flatnonzero(t == row.time_min)[0]), int(row.cell_id)
        for k in (1, 7, 30):
            assert row[f'y_lag_{k}'] == sub[key]['y'][cell, max(i - k, 0)]
            assert row[f'u_lag_{k}'] == c.command(np.array([row.time_min - 2 * k]), 'fgf_mixed', sub[key]['concentration'])[0]
    return dict(rows_checked=25, past_only=True)


def nesting_check():
    for h in (2., 10.):
        for window in ax.WINDOWS:
            a, am = set(ax.arx_features(h, window)), set(ax.arx_features(h, window, mechanism=True))
            assert set(nf.nested_features(h)['C']) < a < am and am - a == {'b3_delta'}
    return dict(nested=True)


def main():
    saved = json.loads(ax.RESULTS.read_text(encoding='utf-8'))
    protocol, roles, blocks, population = ax.load_inputs()
    worst = 0.
    for h, analysis in saved['main'].items():
        rows = ax.add_lags(bf.forecast_rows(blocks, population, float(h), [nf.sigma_value(analysis['sigma'])]), blocks)
        data = rows.assign(b3_delta=rows[f'b3_delta_{analysis["sigma"]}'])
        for role, models in analysis['scores'].items():
            part = data[data.run_id.isin(roles[role])]
            for name, metric in models.items():
                worst = max(worst, abs(run_metric(part, predict(part, analysis['models'][name]))['run_equal_mse'] - metric['run_equal_mse']))
    assert worst < 1e-12, worst
    rows10 = ax.add_lags(bf.forecast_rows(blocks, population, 10., protocol['selection_sigma_grid']), blocks)
    groups = nf.lopo_groups(blocks, roles)
    held = 'fgf_sp_10'
    base = ax.lopo_fold(rows10, groups, held, protocol)
    shifted = rows10.copy()
    shifted.loc[shifted.run_id.isin(groups[held]), 'target'] += 1.
    moved = ax.lopo_fold(shifted, groups, held, protocol)
    assert base['sigma'] == moved['sigma'] and base['window'] == moved['window'] and base['alpha'] == moved['alpha']
    for name in base['prediction']:
        np.testing.assert_allclose(base['prediction'][name], moved['prediction'][name], atol=1e-12)
    report = dict(status='passed', protocol_sha256=ax.protocol_sha256(), lags=lag_check(blocks), nesting=nesting_check(),
                  scores_recomputed=sum(len(m) for a in saved['main'].values() for m in a['scores'].values()),
                  max_abs_score_difference=worst,
                  lopo_leakage=dict(held_out=held, selection_unchanged=True, predictions_unchanged=True))
    (ax.OUT / 'nested_arx_verification_results.json').write_text(json.dumps(report, indent=2) + '\n', newline='\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    sys.exit(main())
