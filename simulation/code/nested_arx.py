"""Lagged non-mechanistic history baseline (ARX) for the nested FGF2-ERK forecast comparison.

Implements configs/nested_arx_protocol.json (reviewer item R2), frozen before
its only run: A adds reporter and command lags to the current-state set C,
with a training-selected window; A+M adds the matched-feedback B3 increment.
Post hoc robustness analysis on inspected roles, sharing data roles, origins,
future-command features, scores and B3 increments with nested_forecast.py.
"""
import hashlib
import json

import numpy as np

import b3_forecast as bf
import calibrate_fgf2 as c
import expanded_fgf2 as e
import nested_forecast as nf
from history_baselines import fit_ridge, predict, run_metric

PROTOCOL = nf.ROOT / 'configs/nested_arx_protocol.json'
FREEZE = nf.OUT / 'nested_arx_protocol_freeze.json'
RESULTS = nf.OUT / 'nested_arx_results.json'
OUT = nf.OUT
WINDOWS = (5, 15, 30)
MODELS = ('A', 'A+M')


def protocol_sha256():
    return hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()


def load_inputs():
    if protocol_sha256() != json.loads(FREEZE.read_text(encoding='utf-8'))['protocol_sha256']:
        raise SystemExit('Protocol differs from its freeze record; refusing to run.')
    protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    nested_protocol, roles, blocks, population = nf.load_inputs()
    protocol['selection_sigma_grid'] = nested_protocol['selection']['sigma_grid']
    return protocol, roles, blocks, population


def add_lags(rows, blocks, max_lag=max(WINDOWS)):
    """Reporter lags y(t - 2k) (first frame repeated before the start) and command lags u(t - 2k) (zero before the origin)."""
    ylag, ulag = np.zeros((len(rows), max_lag)), np.zeros((len(rows), max_lag))
    run, cell_ids, times = rows.run_id.to_numpy(), rows.cell_id.to_numpy(), rows.time_min.to_numpy()
    for key, block in blocks.items():
        mask = run == key
        if not mask.any():
            continue
        t = block['time']
        idx = np.searchsorted(t, times[mask])
        assert np.array_equal(t[idx], times[mask])
        cells = cell_ids[mask].astype(int)
        for k in range(1, max_lag + 1):
            ylag[mask, k - 1] = block['y'][cells, np.maximum(idx - k, 0)]
            ulag[mask, k - 1] = c.command(t[idx] - 2. * k, block['protocol'], block['concentration'])
    out = rows.copy()
    for k in range(max_lag):
        out[f'y_lag_{k + 1}'] = ylag[:, k]
        out[f'u_lag_{k + 1}'] = ulag[:, k]
    return out


def arx_features(horizon, window, mechanism=False):
    cols = nf.nested_features(horizon)['C'] + [f'y_lag_{k}' for k in range(1, window + 1)] + [f'u_lag_{k}' for k in range(1, window + 1)]
    return cols + (['b3_delta'] if mechanism else [])


def select_fixed(train, validation, horizon, mechanism, alphas):
    best = None
    for window in WINDOWS:
        score, model = e.fit_selected(train, validation, arx_features(horizon, window, mechanism), alphas)
        if best is None or score < best[0]:
            best = (score, window, model)
    return best


def comparisons(scores, nested_scores):
    out = {}
    for role, models in scores.items():
        both = dict(nested_scores[role], **models)
        rmse = {n: m['run_equal_mse'] ** .5 for n, m in both.items()}
        runs = both['P']['per_run']
        wins = lambda a, b: int(sum(both[a]['per_run'][k]['mse'] < both[b]['per_run'][k]['mse'] for k in runs))
        out[role] = dict(rmse={n: rmse[n] for n in models}, n_conditions=len(runs),
                         **{f'{a}_beats_{b}': dict(holds=bool(rmse[a] < rmse[b]), conditions=wins(a, b),
                                                   relative_reduction=1 - rmse[a] / rmse[b])
                            for a, b in (('A', 'O'), ('A', 'P'), ('A+M', 'A'), ('A+M', 'H'), ('H', 'A'), ('M', 'A'))})
    return out


def main_analysis(rows, roles, protocol, horizon, nested_main):
    sigma = nested_main['sigma']
    data = rows.assign(b3_delta=rows[f'b3_delta_{sigma}'])
    train, validation = data[data.run_id.isin(roles['train'])], data[data.run_id.isin(roles['validation'])]
    models, selection = {}, {}
    for name in MODELS:
        score, window, model = select_fixed(train, validation, horizon, name == 'A+M', protocol['selection']['ridge_alphas'])
        models[name], selection[name] = model, dict(window=window, alpha=model['alpha'], validation_condition_equal_mse=score)
    scores = {}
    for role in e.SCORED_ROLES:
        part = data[data.run_id.isin(roles[role])]
        if not part.empty:
            scores[role] = {n: run_metric(part, predict(part, m)) for n, m in models.items()}
    return dict(sigma=sigma, selection=selection, models=models, scores=scores,
                comparisons=comparisons(scores, nested_main['scores'])), data


def phase_scores(data, roles, models, horizon):
    keys = [k for r in e.NEW_TEST_ROLES for k in roles[r]]
    test = data[data.run_id.isin(keys)].copy()
    test['phase'] = [nf.phase(t, horizon, p) for t, p in zip(test.time_min, test.history_id)]
    preds = {n: predict(test, m) for n, m in models.items()}
    pooled, detail = {}, {}
    for name in nf.PHASES:
        mask = (test.phase == name).to_numpy()
        if not mask.any():
            continue
        part = test[mask]
        per_model = {m: run_metric(part, yp[mask]) for m, yp in preds.items()}
        pooled[name] = {m: v['run_equal_mse'] ** .5 for m, v in per_model.items()}
        detail[name] = {key: dict(rows=int((part.run_id == key).sum()), cells=int(part[part.run_id == key].cell_id.nunique()),
                                  origins=int(part[part.run_id == key].time_min.nunique()),
                                  rmse={m: float(np.sqrt(v['per_run'][key]['mse'])) for m, v in per_model.items()})
                        for key in sorted(part.run_id.unique())}
    return dict(pooled=pooled, per_condition=detail)


def inner_select(data, groups, train_groups, horizon, mechanism, alphas):
    best = None
    for window in WINDOWS:
        cols = arx_features(horizon, window, mechanism)
        for alpha in alphas:
            losses = []
            for inner in train_groups:
                fit = data[data.run_id.isin([k for g in train_groups if g != inner for k in groups[g]])]
                val = data[data.run_id.isin(groups[inner])]
                losses.append(run_metric(val, predict(val, fit_ridge(fit, cols, alpha)))['run_equal_mse'])
            score = float(np.mean(losses))
            if best is None or score < best[0]:
                best = (score, window, alpha)
    return best[1], best[2]


def lopo_fold(rows, groups, held, protocol):
    train_groups = [g for g in groups if g != held]
    train_keys = [k for g in train_groups for k in groups[g]]
    sigma, _ = nf.select_sigma(rows[rows.run_id.isin(train_keys)], protocol['selection_sigma_grid'])
    data = rows.assign(b3_delta=rows[f'b3_delta_{sigma}'])
    train, test = data[data.run_id.isin(train_keys)], data[data.run_id.isin(groups[held])]
    assert not set(train.run_id) & set(test.run_id)
    fold = dict(conditions=groups[held], sigma=sigma, window={}, alpha={}, prediction={})
    for name in MODELS:
        window, alpha = inner_select(data, groups, train_groups, 10., name == 'A+M', protocol['selection']['ridge_alphas'])
        fold['window'][name], fold['alpha'][name] = window, alpha
        fold['prediction'][name] = predict(test, fit_ridge(train, arx_features(10., window, name == 'A+M'), alpha))
    fold['rmse'] = {n: run_metric(test, yp)['run_equal_mse'] ** .5 for n, yp in fold['prediction'].items()}
    return fold


def main():
    protocol, roles, blocks, population = load_inputs()
    nested = json.loads(nf.RESULTS.read_text(encoding='utf-8'))
    main_out, phases = {}, {}
    for horizon in protocol['horizons_min']:
        h = f'{horizon:g}'
        rows = add_lags(bf.forecast_rows(blocks, population, float(horizon), protocol['selection_sigma_grid']), blocks)
        main_out[h], data = main_analysis(rows, roles, protocol, float(horizon), nested['main'][h])
        phases[h] = phase_scores(data, roles, main_out[h]['models'], float(horizon))
        if h == '10':
            groups = nf.lopo_groups(blocks, roles)
            lopo = {}
            for held in groups:
                fold = lopo_fold(rows, groups, held, protocol)
                fold.pop('prediction')
                fold['nested_rmse'] = nested['lopo'][held]['rmse']
                lopo[held] = fold
    result = dict(status='complete', protocol_sha256=protocol_sha256(),
                  frozen_utc=json.loads(FREEZE.read_text(encoding='utf-8'))['frozen_utc'],
                  main=main_out, phases=phases, lopo=lopo,
                  limitations=['Post hoc robustness analysis on inspected roles; not a new confirmation.',
                               'B3 is a fixed, previously calibrated reference (see nested_results.json b3_roles).'])
    RESULTS.write_text(json.dumps(result, indent=2) + '\n', newline='\n')
    print(json.dumps(dict(selection={h: a['selection'] for h, a in main_out.items()},
                          rmse={h: {r: {k: round(v, 5) for k, v in comp['rmse'].items()} for r, comp in a['comparisons'].items()}
                                for h, a in main_out.items()},
                          counts={r: {k: comp[k]['conditions'] for k in comp if '_beats_' in k} for r, comp in main_out['10']['comparisons'].items()},
                          phases={p: {k: round(v, 5) for k, v in r.items()} for p, r in phases['10']['pooled'].items()},
                          next_command=phases['10']['per_condition'].get('next_command'),
                          lopo={g: dict(rmse={k: round(v, 5) for k, v in f['rmse'].items()}, window=f['window']) for g, f in lopo.items()}),
                     indent=1))


if __name__ == '__main__':
    main()
