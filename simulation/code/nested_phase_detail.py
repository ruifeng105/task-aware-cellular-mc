"""Per-condition forecast errors by phase for the frozen nested models.

Descriptive breakdown of results/nested/nested_results.json: no selection and
no fitting. For every test condition and forecast phase it reports the number
of forecast rows (origins x cells), cells, origin times and the 10-min RMSE of
P, C, O, M and H. Pooling the per-condition MSEs with equal weights
reproduces the frozen phase scores (checked here and in
verify_nested_forecast.py).
"""
import json

import numpy as np

import b3_forecast as bf
import expanded_fgf2 as e
import nested_forecast as nf
from history_baselines import run_metric

DETAIL = nf.OUT / 'phase_detail.json'
HORIZON = 10.


def detail(data, roles, models):
    keys = [k for r in e.NEW_TEST_ROLES for k in roles[r]]
    test = data[data.run_id.isin(keys)].copy()
    test['phase'] = [nf.phase(t, HORIZON, p) for t, p in zip(test.time_min, test.history_id)]
    preds = nf.predictions(test, models, names=nf.MODELS)
    out = {}
    for name in nf.PHASES:
        mask = (test.phase == name).to_numpy()
        if not mask.any():
            continue
        part = test[mask]
        per_model = {m: run_metric(part, yp[mask])['per_run'] for m, yp in preds.items()}
        out[name] = {}
        for key in sorted(part.run_id.unique()):
            rows = part[part.run_id == key]
            out[name][key] = dict(rows=int(len(rows)), cells=int(rows.cell_id.nunique()),
                                  origins=int(rows.time_min.nunique()),
                                  origin_range_min=[float(rows.time_min.min()), float(rows.time_min.max())],
                                  rmse={m: float(np.sqrt(per_model[m][key]['mse'])) for m in preds})
    return out


def main():
    protocol, roles, blocks, population = nf.load_inputs()
    saved = json.loads(nf.RESULTS.read_text(encoding='utf-8'))
    analysis = saved['main']['10']
    rows = bf.forecast_rows(blocks, population, HORIZON, [nf.sigma_value(analysis['sigma'])])
    data = rows.assign(b3_delta=rows[f'b3_delta_{analysis["sigma"]}'])
    phases = detail(data, roles, analysis['models'])
    for phase, conditions in phases.items():
        for name, value in saved['phases']['10'][phase]['rmse'].items():
            pooled = float(np.sqrt(np.mean([c['rmse'][name] ** 2 for c in conditions.values()])))
            assert abs(pooled - value) < 1e-12, (phase, name, pooled, value)
    result = dict(status='complete', source_sha256=nf.hashlib.sha256(nf.RESULTS.read_bytes()).hexdigest(),
                  horizon_min=HORIZON, phases=phases,
                  scope='Descriptive breakdown of the frozen nested models; post hoc on inspected roles.')
    DETAIL.write_text(json.dumps(result, indent=2) + '\n', newline='\n')
    print(json.dumps({key: {k: v for k, v in row.items() if k != 'origin_range_min'}
                      for key, row in phases['next_command'].items()}, indent=2))


if __name__ == '__main__':
    main()
