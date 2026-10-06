"""Nested P/C/O/M/H forecast comparison, phase-wise scores and corrected LOPO.

Implements configs/nested_forecast_protocol.json, a post hoc robustness
analysis on already inspected roles. Refuses to run if the protocol differs
from its freeze record or the B3 acceptance check has not passed. C uses the
current input, reporter, slope and known future commands; O adds causal
reporter filters; M adds the matched-feedback B3 increment; H adds input
filters and cumulative exposure (the feature set of M3 in b3_forecast.py).
"""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import b3_forecast as bf
import calibrate_fgf2 as c
import expanded_fgf2 as e
from history_baselines import fit_ridge, predict, run_metric

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'configs/nested_forecast_protocol.json'
OUT = ROOT / 'results/nested'
FREEZE = OUT / 'nested_forecast_protocol_freeze.json'
RESULTS = OUT / 'nested_results.json'
B3_CONFIG = ROOT / 'public_data/fgf2_author_repository/FGF2_models/Fgf_B3/config_fgf_sus_3_20.xml'
MODELS = ('C', 'O', 'M', 'H')
PHASES = ('stimulation', 'next_command', 'early_washout', 'late')
EARLY_WASHOUT_MIN = 30.
PAIRS = (('H', 'M'), ('M', 'O'), ('O', 'C'), ('C', 'P'), ('H', 'P'))


def protocol_sha256():
    return hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()


def load_protocol():
    if protocol_sha256() != json.loads(FREEZE.read_text(encoding='utf-8'))['protocol_sha256']:
        raise SystemExit('Protocol differs from its freeze record; refusing to run.')
    if json.loads((bf.OUT / 'verification_results.json').read_text(encoding='utf-8'))['status'] != 'passed':
        raise SystemExit('B3 acceptance check has not passed.')
    return json.loads(PROTOCOL.read_text(encoding='utf-8'))


def load_inputs():
    protocol = load_protocol()
    expanded = e.load_protocol()
    blocks, _ = e.load_blocks(expanded)
    return protocol, expanded['roles'], blocks, bf.simulate_population(blocks)


def sigma_value(label):
    return label if label == 'equal_weights' else float(label)


def nested_features(horizon):
    taus = c.HISTORY_TAUS_MIN
    sets = {'C': c.model_features(horizon)['local_slope']}
    sets['O'] = sets['C'] + [f'response_ema_{t:g}' for t in taus]
    sets['M'] = sets['O'] + ['b3_delta']
    sets['H'] = sets['M'] + ['past_dose'] + [f'input_filter_{t:g}' for t in taus]
    sets.update({f'{name}+clock': cols + ['time_min', 'time2'] for name, cols in list(sets.items())})
    return sets


def phase(origin, horizon, protocol):
    """Phase of a forecast from `origin` to `origin + horizon` under the half-open pulse schedule."""
    pulses = c.segments(protocol, until=origin + horizon + 1.)
    if any(a <= origin < b for a, b in pulses):
        return 'stimulation'
    if any(origin < a <= origin + horizon for a, _ in pulses):
        return 'next_command'
    ended = [b for _, b in pulses if b <= origin]
    if not ended:
        raise ValueError(f'forecast origin {origin} precedes every pulse of {protocol}')
    return 'early_washout' if origin - max(ended) <= EARLY_WASHOUT_MIN else 'late'


def select_sigma(rows, sigmas):
    scores = {bf.label(s): run_metric(rows, rows.current_y.to_numpy() + rows[f'b3_delta_{bf.label(s)}'].to_numpy())['run_equal_mse']
              for s in sigmas}
    return min(scores, key=scores.get), scores


def predictions(rows, models, names=None):
    out = {'P': rows.current_y.to_numpy()}
    out.update({n: predict(rows, m) for n, m in models.items() if names is None or n in names})
    return out


def comparisons(scores):
    out = {}
    for role, models in scores.items():
        rmse = {n: m['run_equal_mse'] ** .5 for n, m in models.items()}
        runs = models['P']['per_run']
        wins = lambda a, b: int(sum(models[a]['per_run'][k]['mse'] < models[b]['per_run'][k]['mse'] for k in runs))
        out[role] = dict(rmse=rmse, n_conditions=len(runs),
                         **{f'{a}_beats_{b}': dict(holds=bool(rmse[a] < rmse[b]), conditions=wins(a, b),
                                                   relative_reduction=1 - rmse[a] / rmse[b]) for a, b in PAIRS})
    return out


def main_analysis(rows, roles, protocol, horizon):
    sigma, sigma_scores = select_sigma(rows[rows.run_id.isin(roles['validation'])], protocol['selection']['sigma_grid'])
    data = rows.assign(b3_delta=rows[f'b3_delta_{sigma}'])
    train, validation = data[data.run_id.isin(roles['train'])], data[data.run_id.isin(roles['validation'])]
    models, selection = {}, {}
    for name, cols in nested_features(horizon).items():
        score, model = e.fit_selected(train, validation, cols, protocol['selection']['ridge_alphas'])
        models[name], selection[name] = model, dict(alpha=model['alpha'], validation_condition_equal_mse=score)
    scores = {}
    for role in e.SCORED_ROLES:
        part = data[data.run_id.isin(roles[role])]
        if not part.empty:
            scores[role] = {n: run_metric(part, yp) for n, yp in predictions(part, models).items()}
    return dict(sigma=sigma, sigma_validation_mse=sigma_scores, selection=selection, models=models,
                scores=scores, comparisons=comparisons(scores)), data


def phase_scores(data, roles, models, horizon):
    keys = [k for r in e.NEW_TEST_ROLES for k in roles[r]]
    test = data[data.run_id.isin(keys)].copy()
    test['phase'] = [phase(t, horizon, p) for t, p in zip(test.time_min, test.history_id)]
    preds = predictions(test, models, names=MODELS)
    out = {}
    for name in PHASES:
        mask = (test.phase == name).to_numpy()
        if mask.any():
            part = test[mask]
            out[name] = dict(n_conditions=int(part.run_id.nunique()), n_rows=int(mask.sum()),
                             rows_by_role={r: int(part.run_id.isin(roles[r]).sum()) for r in e.NEW_TEST_ROLES},
                             rmse={m: run_metric(part, yp[mask])['run_equal_mse'] ** .5 for m, yp in preds.items()})
    return out


def example_cell(data, models, blocks, protocol):
    key = protocol['example']['condition']
    part = data[data.run_id == key]
    preds = predictions(part, models, names=('M', 'H'))
    mse = pd.Series((preds['H'] - part.target.to_numpy()) ** 2).groupby(part.cell_id.to_numpy()).mean()
    ordered = sorted(mse.items(), key=lambda kv: (kv[1], int(kv[0])))
    cell = ordered[len(ordered) // 2][0]
    mask = (part.cell_id == cell).to_numpy()
    block = blocks[key]
    target = part.target.to_numpy()[mask]
    return dict(condition=key, cell=int(cell), n_cells=len(ordered), horizon_min=10.,
                time_min=block['time'].tolist(), observed=block['y'][int(cell)].tolist(),
                quartiles=np.percentile(block['y'], [25, 50, 75], axis=0).tolist(),
                pulses=[list(s) for s in c.segments(block['protocol'], until=float(block['time'].max()))],
                target_time_min=(part.time_min.to_numpy()[mask] + 10.).tolist(),
                forecast={n: preds[n][mask].tolist() for n in ('P', 'M', 'H')},
                cell_rmse={n: float(np.sqrt(np.mean((preds[n][mask] - target) ** 2))) for n in ('P', 'M', 'H')})


def b3_data_roles():
    text = B3_CONFIG.read_text(encoding='utf-8')
    grab = lambda block, tag: re.search(rf'<{block}>.*?<{tag}>(.*?)</{tag}>', text, re.S).group(1).split()
    return dict(fitted=grab('LFNS', 'experiments'), architecture_likelihood=grab('ComputeLikelihood', 'modelconfigurations'))


def b3_use(conditions, roles_b3):
    author = [k.replace('fgf_', '', 1) for k in conditions]
    return dict(fitted=[k for k in author if k in roles_b3['fitted']],
                architecture_likelihood=[k for k in author if k in roles_b3['architecture_likelihood']])


def lopo_groups(blocks, roles):
    groups = {}
    for key in [k for r in ('train', 'validation') + e.NEW_TEST_ROLES for k in roles[r] if k in blocks]:
        name = key.rsplit('_', 1)[0]
        groups.setdefault('fgf_mixed_new' if name == 'fgf_mixed' else name, []).append(key)
    return groups


def inner_alpha(data, groups, train_groups, cols, alphas):
    mean = {}
    for alpha in alphas:
        losses = []
        for inner in train_groups:
            fit = data[data.run_id.isin([k for g in train_groups if g != inner for k in groups[g]])]
            val = data[data.run_id.isin(groups[inner])]
            losses.append(run_metric(val, predict(val, fit_ridge(fit, cols, alpha)))['run_equal_mse'])
        mean[alpha] = float(np.mean(losses))
    return min(mean, key=mean.get)


def lopo_fold(rows, groups, held, protocol):
    train_groups = [g for g in groups if g != held]
    train_keys = [k for g in train_groups for k in groups[g]]
    sigma, _ = select_sigma(rows[rows.run_id.isin(train_keys)], protocol['selection']['sigma_grid'])
    data = rows.assign(b3_delta=rows[f'b3_delta_{sigma}'])
    train, test = data[data.run_id.isin(train_keys)], data[data.run_id.isin(groups[held])]
    assert not set(train.run_id) & set(test.run_id)
    fold = dict(conditions=groups[held], sigma=sigma, alpha={}, prediction={'P': test.current_y.to_numpy()})
    features = nested_features(10.)
    for name in MODELS:
        alpha = inner_alpha(data, groups, train_groups, features[name], protocol['selection']['ridge_alphas'])
        fold['alpha'][name] = alpha
        fold['prediction'][name] = predict(test, fit_ridge(train, features[name], alpha))
    fold['rmse'] = {n: run_metric(test, yp)['run_equal_mse'] ** .5 for n, yp in fold['prediction'].items()}
    return fold


def corrected_lopo(rows, blocks, roles, protocol, roles_b3):
    groups = lopo_groups(blocks, roles)
    out = {}
    for held in groups:
        fold = lopo_fold(rows, groups, held, protocol)
        fold.pop('prediction')
        fold['b3_use'] = b3_use(groups[held], roles_b3)
        out[held] = fold
    return out


def decision(main10, lopo):
    primary = main10['comparisons']['test_new_protocol']['H_beats_M']['holds']
    folds = int(sum(f['rmse']['H'] < f['rmse']['M'] for f in lopo.values()))
    return dict(H_beats_M_test_A=primary, H_beats_M_lopo_folds=folds, n_folds=len(lopo),
                input_history_reported_as_contribution=bool(primary and folds >= 4))


def main():
    protocol, roles, blocks, population = load_inputs()
    roles_b3 = b3_data_roles()
    main_out, phases = {}, {}
    for horizon in protocol['horizons_min']:
        h = f'{horizon:g}'
        rows = bf.forecast_rows(blocks, population, float(horizon), protocol['selection']['sigma_grid'])
        main_out[h], data = main_analysis(rows, roles, protocol, float(horizon))
        phases[h] = phase_scores(data, roles, main_out[h]['models'], float(horizon))
        if h == '10':
            example = example_cell(data, main_out[h]['models'], blocks, protocol)
            lopo = corrected_lopo(rows, blocks, roles, protocol, roles_b3)
    result = dict(status='complete', protocol_sha256=protocol_sha256(),
                  frozen_utc=json.loads(FREEZE.read_text(encoding='utf-8'))['frozen_utc'],
                  main=main_out, phases=phases, example=example, lopo=lopo, b3_roles=roles_b3,
                  decision=decision(main_out['10'], lopo),
                  limitations=['Post hoc robustness analysis on inspected roles; not a new confirmation.',
                               'Conditions are not verified independent replicates; protocol generalization is not cross-day generalization.',
                               'B3 is a calibrated reference whose fit or architecture comparison used some held-out protocols (see b3_use).'])
    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(result, indent=2) + '\n', newline='\n')
    print(json.dumps(dict(rmse={h: {r: {k: round(v, 5) for k, v in comp['rmse'].items()} for r, comp in a['comparisons'].items()}
                                for h, a in main_out.items()},
                          lopo={g: {k: round(v, 5) for k, v in f['rmse'].items()} for g, f in lopo.items()},
                          decision=result['decision']), indent=2))


if __name__ == '__main__':
    main()
