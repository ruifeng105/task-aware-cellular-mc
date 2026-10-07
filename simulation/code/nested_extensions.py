"""Longer horizons, noise-adjusted skill, crossing score, cell bootstrap and phase-gated B3 increments.

Implements configs/nested_extensions_protocol.json (revision plan items 6, 7
and 9), a post hoc analysis on inspected roles frozen before its only run.
The nested predictors P/C/O/M/H are refitted exactly as in
nested_forecast.py (horizons 2 and 10 reproduce the frozen results) and
extended to 20 and 30 min. Scores add a white-noise floor estimated from
second differences, the balanced accuracy of anticipating the half-decay
crossing, and stratified cell-bootstrap intervals. Two phase-gated variants
of M, fitted on the training protocols only, address the next-command
windows in which the B3 increment raised the error.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import b3_forecast as bf
import calibrate_fgf2 as c
import expanded_fgf2 as e
import nested_forecast as nf
from history_baselines import fit_ridge, predict, run_metric

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'configs/nested_extensions_protocol.json'
FREEZE = nf.OUT / 'nested_extensions_protocol_freeze.json'
RESULTS = nf.OUT / 'nested_extensions_results.json'
HORIZONS = (2., 10., 20., 30.)
MODELS = ('P', 'C', 'O', 'M', 'H', 'M_phase', 'M_switch')
GATED_TERMS = ('b3_delta', 'current_y', 'slope')
BOOTSTRAP, BOOT_SEED = 2000, 20261009
REDUCTIONS = (('O', 'C'), ('M', 'O'), ('H', 'M'), ('M', 'P'), ('M_phase', 'M'))
SKILL_MODELS = ('C', 'O', 'M', 'H', 'M_phase')
CROSSING_HORIZONS = (10., 20., 30.)


def protocol_sha256():
    return hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()


def load_protocol():
    if protocol_sha256() != json.loads(FREEZE.read_text(encoding='utf-8'))['protocol_sha256']:
        raise SystemExit('Protocol differs from its freeze record; refusing to run.')
    return json.loads(PROTOCOL.read_text(encoding='utf-8'))


def with_phases(rows, horizon):
    rows = rows.copy()
    rows['phase'] = [nf.phase(t, horizon, p) for t, p in zip(rows.time_min, rows.history_id)]
    for name, phase in (('st', 'stimulation'), ('nc', 'next_command')):
        rows[name] = (rows.phase == phase).astype(float)
        for term in GATED_TERMS:
            rows[f'{name}_{term}'] = rows[name] * rows[term]
    return rows


def phase_features(horizon):
    base = nf.nested_features(horizon)['M']
    return base + [f'{p}{suffix}' for p in ('st', 'nc') for suffix in [''] + [f'_{t}' for t in GATED_TERMS]]


def switch_gate(train, models):
    """'O' if O has the lower condition-equal in-sample MSE than M on next-command training rows, else 'M'."""
    part = train[train.phase == 'next_command']
    if part.empty:
        return 'M', None
    mse = {n: run_metric(part, predict(part, models[n]))['run_equal_mse'] for n in ('O', 'M')}
    return ('O' if mse['O'] < mse['M'] else 'M'), mse


def all_predictions(rows, models, gate):
    out = {'P': rows.current_y.to_numpy()}
    for name in ('C', 'O', 'M', 'H', 'M_phase'):
        out[name] = predict(rows, models[name])
    out['M_switch'] = np.where((rows.phase == 'next_command').to_numpy(), out[gate], out['M'])
    return out


def noise_floor(blocks):
    """Robust second-difference estimate of white reporter noise per condition."""
    out = {}
    for key, block in blocks.items():
        y = np.asarray(block['y'], dtype=float)
        d2 = (y[:, 2:] - 2 * y[:, 1:-1] + y[:, :-2]).ravel()
        d2 = d2[np.isfinite(d2)]
        out[key] = float(np.median(np.abs(d2)) / (0.6745 * np.sqrt(6.)))
    return out


def skill(per_run, persistence, floor):
    num = np.mean([per_run[k] - floor[k] ** 2 for k in per_run])
    den = np.mean([persistence[k] - floor[k] ** 2 for k in persistence])
    return float(1 - num / den)


def excursion_peaks(blocks):
    """Running maximum of y - 1 since the onset of the most recent pulse, per cell and frame."""
    out = {}
    for key, block in blocks.items():
        t = block['time']
        onsets = [a for a, _ in c.segments(block['protocol'], until=float(t.max()) + 1.)]
        last = np.array([max([a for a in onsets if a <= now], default=-np.inf) for now in t])
        for cell, y in enumerate(block['y']):
            excursion = np.asarray(y, dtype=float) - 1.
            peak = np.empty_like(excursion)
            for i in range(len(t)):
                window = (t >= last[i]) & (t <= t[i])
                peak[i] = np.nanmax(excursion[window]) if window.any() else np.nan
            out[(key, str(cell))] = (t, peak)
    return out


def crossing_rows(rows, peaks):
    """Undecided washout rows with their running peak; returns mask and peak array."""
    peak = np.full(len(rows), np.nan)
    for i, (run, cell, now) in enumerate(zip(rows.run_id, rows.cell_id, rows.time_min)):
        t, p = peaks[(run, cell)]
        peak[i] = p[int(np.searchsorted(t, now))]
    washout = rows.phase.isin(('early_washout', 'late')).to_numpy()
    undecided = (rows.current_y.to_numpy() - 1.) > .5 * peak
    return washout & undecided & np.isfinite(peak), peak


def crossing_score(rows, prediction, mask, peak):
    """Condition-equal balanced accuracy of anticipating the half-decay crossing."""
    scores = {}
    event = (rows.target.to_numpy() - 1.) <= .5 * peak
    guess = (prediction - 1.) <= .5 * peak
    for run in rows.run_id[mask].unique():
        m = mask & (rows.run_id == run).to_numpy()
        pos, neg = event[m], ~event[m]
        if pos.any() and neg.any():
            scores[run] = .5 * (guess[m][pos].mean() + (~guess[m][neg]).mean())
    return dict(balanced_accuracy=float(np.mean(list(scores.values()))) if scores else None,
                n_conditions=len(scores), per_run={k: float(v) for k, v in scores.items()},
                n_rows=int(mask.sum()))


def bootstrap(part, preds, floor, rng):
    """Stratified cell bootstrap of reductions and skills for one role (models and floors fixed)."""
    conditions = sorted(part.run_id.unique())
    draws = {}
    for run in conditions:
        m = (part.run_id == run).to_numpy()
        cells = part.cell_id.to_numpy()[m]
        ids, inverse = np.unique(cells, return_inverse=True)
        target = part.target.to_numpy()[m]
        sse = np.stack([np.bincount(inverse, (preds[n][m] - target) ** 2, len(ids)) for n in MODELS], 1)
        count = np.bincount(inverse, minlength=len(ids)).astype(float)
        index = rng.integers(0, len(ids), (BOOTSTRAP, len(ids)))
        draws[run] = sse[index].sum(1) / count[index].sum(1)[:, None]          # (B, models)
    mse = np.mean([draws[r] for r in conditions], axis=0)                       # condition-equal
    adjusted = np.mean([draws[r] - floor[r] ** 2 for r in conditions], axis=0)
    col = {n: i for i, n in enumerate(MODELS)}
    ci = lambda v: np.percentile(v, [2.5, 97.5]).tolist()
    out = {f'{a}_vs_{b}': ci(1 - np.sqrt(mse[:, col[a]] / mse[:, col[b]])) for a, b in REDUCTIONS}
    out.update({f'skill_{n}': ci(1 - adjusted[:, col[n]] / adjusted[:, col['P']]) for n in SKILL_MODELS})
    return out


def horizon_block(rows, roles, nested_protocol, horizon, floor, peaks, frozen, rng):
    out, data = nf.main_analysis(rows, roles, nested_protocol, horizon)
    data = with_phases(data, horizon)
    train, validation = data[data.run_id.isin(roles['train'])], data[data.run_id.isin(roles['validation'])]
    models = dict(out['models'])
    score, models['M_phase'] = e.fit_selected(train, validation, phase_features(horizon), nested_protocol['selection']['ridge_alphas'])
    gate, gate_mse = switch_gate(train, models)
    block = dict(sigma=out['sigma'], alpha={n: models[n]['alpha'] for n in ('C', 'O', 'M', 'H', 'M_phase')},
                 m_phase_validation_mse=score, switch_gate=gate, switch_training_next_command_mse=gate_mse,
                 roles={}, phases={})
    if frozen is not None:
        block['reproduction_max_abs_rmse_difference'] = max(
            abs(out['comparisons'][role]['rmse'][n] - frozen['comparisons'][role]['rmse'][n])
            for role in frozen['comparisons'] for n in ('P', 'C', 'O', 'M', 'H'))
    test_keys = [k for r in e.NEW_TEST_ROLES for k in roles[r]]
    for role in ('validation',) + e.NEW_TEST_ROLES:
        part = data[data.run_id.isin(roles[role])]
        preds = all_predictions(part, models, gate)
        metrics = {n: run_metric(part, p) for n, p in preds.items()}
        per_run = {n: {k: v['mse'] for k, v in m['per_run'].items()} for n, m in metrics.items()}
        rmse = {n: m['run_equal_mse'] ** .5 for n, m in metrics.items()}
        adjusted = {n: float(np.mean([per_run[n][k] - floor[k] ** 2 for k in per_run[n]])) for n in MODELS}
        role_out = dict(rmse=rmse, per_run_mse=per_run,
                        skill={n: skill(per_run[n], per_run['P'], floor) for n in MODELS if n != 'P'},
                        noise_fraction=float(np.mean([floor[k] ** 2 / per_run['P'][k] for k in per_run['P']])),
                        noise_adjusted_reduction={f'{a}_vs_{b}': 1 - adjusted[a] / adjusted[b] for a, b in REDUCTIONS},
                        reduction={f'{a}_vs_{b}': 1 - rmse[a] / rmse[b] for a, b in REDUCTIONS},
                        beats={f'{a}_vs_{b}': int(sum(per_run[a][k] < per_run[b][k] for k in per_run[a])) for a, b in REDUCTIONS},
                        n_conditions=len(per_run['P']))
        if horizon in CROSSING_HORIZONS:
            mask, peak = crossing_rows(part, peaks)
            role_out['crossing'] = {n: crossing_score(part, p, mask, peak) for n, p in preds.items()}
        if role != 'validation':
            role_out['bootstrap'] = bootstrap(part, preds, floor, rng)
        block['roles'][role] = role_out
    test = data[data.run_id.isin(test_keys)]
    preds = all_predictions(test, models, gate)
    for name in nf.PHASES:
        mask = (test.phase == name).to_numpy()
        if mask.any():
            part = test[mask]
            block['phases'][name] = dict(n_conditions=int(part.run_id.nunique()), n_rows=int(mask.sum()),
                                         rmse={n: run_metric(part, p[mask])['run_equal_mse'] ** .5 for n, p in preds.items()})
    return block, data


def lopo(rows, blocks, roles, nested_protocol, frozen):
    groups = nf.lopo_groups(blocks, roles)
    features = dict(nf.nested_features(10.), M_phase=phase_features(10.))
    alphas = nested_protocol['selection']['ridge_alphas']
    out = {}
    for held in groups:
        train_groups = [g for g in groups if g != held]
        train_keys = [k for g in train_groups for k in groups[g]]
        sigma, _ = nf.select_sigma(rows[rows.run_id.isin(train_keys)], nested_protocol['selection']['sigma_grid'])
        data = with_phases(rows.assign(b3_delta=rows[f'b3_delta_{sigma}']), 10.)
        train, test = data[data.run_id.isin(train_keys)], data[data.run_id.isin(groups[held])]
        fold = dict(sigma=sigma, alpha={}, rmse={}, next_command={})
        nc = (test.phase == 'next_command').to_numpy()
        for name in ('O', 'M', 'M_phase'):
            alpha = nf.inner_alpha(data, groups, train_groups, features[name], alphas)
            prediction = predict(test, fit_ridge(train, features[name], alpha))
            fold['alpha'][name] = alpha
            fold['rmse'][name] = run_metric(test, prediction)['run_equal_mse'] ** .5
            if nc.any():
                fold['next_command'][name] = run_metric(test[nc], prediction[nc])['run_equal_mse'] ** .5
        fold['reproduction_max_abs_difference'] = max(abs(fold['rmse'][n] - frozen[held]['rmse'][n]) for n in ('O', 'M'))
        out[held] = fold
    return out


def main():
    load_protocol()
    nested_protocol, roles, blocks, population = nf.load_inputs()
    frozen = json.loads(nf.RESULTS.read_text(encoding='utf-8'))
    floor = noise_floor(blocks)
    peaks = excursion_peaks(blocks)
    rng = np.random.default_rng(BOOT_SEED)
    horizons = {}
    for horizon in HORIZONS:
        h = f'{horizon:g}'
        rows = bf.forecast_rows(blocks, population, horizon, nested_protocol['selection']['sigma_grid'])
        horizons[h], _ = horizon_block(rows, roles, nested_protocol, horizon, floor, peaks, frozen['main'].get(h), rng)
        if h == '10':
            folds = lopo(rows, blocks, roles, nested_protocol, frozen['lopo'])
    result = dict(status='complete', protocol_sha256=protocol_sha256(),
                  frozen_utc=json.loads(FREEZE.read_text(encoding='utf-8'))['frozen_utc'],
                  noise_floor=floor, horizons=horizons, lopo=folds,
                  limitations=['Post hoc analysis on inspected roles; not a new confirmation.',
                               'The noise floor assumes white reporter noise; slow measurement drift is counted as signal.',
                               'Bootstrap intervals resample cells within conditions that share a session; they are not experiment-level intervals.',
                               'The phase-gated variants were motivated by the observed next-command result.'])
    RESULTS.write_text(json.dumps(result, indent=2) + '\n', newline='\n')
    for h, block in horizons.items():
        print(h, 'gate', block['switch_gate'], 'repro', block.get('reproduction_max_abs_rmse_difference'))
        for role, r in block['roles'].items():
            print('  ', role, {n: round(v, 5) for n, v in r['rmse'].items()},
                  'skill', {n: round(v, 3) for n, v in r['skill'].items()}, 'noise', round(r['noise_fraction'], 3))
        print('   phases', {p: {n: round(v, 5) for n, v in b['rmse'].items()} for p, b in block['phases'].items()})
    print('lopo', {g: ({n: round(v, 5) for n, v in f['rmse'].items()}, {n: round(v, 5) for n, v in f['next_command'].items()})
                   for g, f in folds.items()})


if __name__ == '__main__':
    main()
