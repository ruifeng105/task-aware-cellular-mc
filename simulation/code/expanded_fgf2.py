"""Pre-registered fresh-test analysis on additional author-exported conditions.

Implements configs/fgf2_expanded_protocol.json and refuses to run when the
protocol differs from its freeze record. Prediction only: no controller,
no new experiment and no independent-day inference. The pilot functions in
calibrate_fgf2.py supply every feature, schedule and score definition.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

import calibrate_fgf2 as c
from history_baselines import fit_ridge, predict, run_metric

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'configs/fgf2_expanded_protocol.json'
OUT = ROOT / 'results/fgf2_expanded'
FREEZE = OUT / 'protocol_freeze.json'
RESULTS = OUT / 'expanded_results.json'
NEW_TEST_ROLES = ('test_new_protocol', 'test_new_protocol_inferred_timing', 'test_new_concentration_same_session')
SCORED_ROLES = ('validation',) + NEW_TEST_ROLES + ('viewed_reference',)
LOPO_MODELS = ('persistence', 'current_response_clock', 'causal_history', 'history_clock')
SP60_SHIFTS = {'onset-1': 'fgf_sp_60_onset-1', 'onset+1': 'fgf_sp_60_onset+1'}


def protocol_sha256():
    return hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()


def load_protocol():
    frozen = json.loads(FREEZE.read_text(encoding='utf-8'))
    if protocol_sha256() != frozen['protocol_sha256']:
        raise SystemExit('Protocol differs from its freeze record; refusing to run.')
    return json.loads(PROTOCOL.read_text(encoding='utf-8'))


def load_blocks(protocol):
    """Apply the protocol's data rules; return included blocks and an inventory."""
    tokens = protocol['concentration_tokens_ng_ml']
    blocks, inventory = {}, []
    for role, conditions in protocol['roles'].items():
        for key in conditions:
            name, token = key.rsplit('_', 1)
            folder = c.DATA / name
            t = c.load_matrix(folder / 'time_trunc.txt').ravel()
            y = c.load_matrix(folder / f'{token}_trunc.txt')
            author_mean = c.load_matrix(folder / f'{token}_mean_trunc.txt').ravel()
            assert y.shape[1] == len(t) == len(author_mean), key
            finite = np.isfinite(y).all(axis=1)
            uniform = bool(np.allclose(np.diff(t), 2.))
            record = dict(condition_id=key, role=role, concentration_ng_ml=tokens[token],
                          n_cells_exported=int(len(y)), n_cells_excluded_nonfinite=int((~finite).sum()),
                          n_cells=int(finite.sum()), n_times=int(len(t)),
                          time_min=[float(t[0]), float(t[-1])], uniform_2min=uniform, included=uniform)
            if finite.any():
                record['author_mean_max_difference'] = float(np.abs(y[finite].mean(0) - author_mean).max())
            inventory.append(record)
            if uniform:
                blocks[key] = dict(protocol=name, token=token, concentration=tokens[token], time=t, y=y[finite])
    return blocks, inventory


def mixed_normalization(protocol):
    """Data rule: rebuild the new mixed concentrations from unnormalized exports."""
    directory = c.DATA / 'fgf_mixed'
    t = c.load_matrix(directory / 'time.txt').ravel()
    start, baseline = 21, (t >= 10.) & (t <= 30.)  # the pilot's rule (MATLAB index 22)
    out = {}
    for key in protocol['roles']['test_new_concentration_same_session']:
        token = key.rsplit('_', 1)[1]
        raw = c.load_matrix(directory / f'{token}_unnormalized.txt')
        supplied = c.load_matrix(directory / f'{token}_trunc.txt')
        if raw.shape != (supplied.shape[0], len(t)) or len(t) - start != supplied.shape[1]:
            out[key] = dict(reconstructable=False, unnormalized_shape=list(raw.shape),
                            supplied_shape=list(supplied.shape))
            continue
        rebuilt = raw[:, start:] / np.median(raw[:, baseline], axis=1)[:, None]
        out[key] = dict(reconstructable=True, max_abs_error=float(np.nanmax(np.abs(rebuilt - supplied))))
    return out


def feature_sets(horizon):
    sets = c.model_features(horizon)
    sets['history_clock'] = sets['causal_history'] + ['time_min', 'time2']
    return sets


def fit_selected(train, validation, features, alphas):
    candidates = []
    for alpha in alphas:
        model = fit_ridge(train, features, alpha)
        candidates.append((run_metric(validation, predict(validation, model))['run_equal_mse'], model))
    return min(candidates, key=lambda item: item[0])


def prediction(rows, model):
    return rows.current_y.to_numpy() if model is None else predict(rows, model)


def horizon_analysis(blocks, protocol, horizon):
    roles = protocol['roles']
    rows = c.forecasting_rows(blocks, horizon)
    train = rows[rows.run_id.isin(roles['train'])]
    validation = rows[rows.run_id.isin(roles['validation'])]
    models = {'persistence': None}
    selection = {'persistence': dict(alpha=None, validation_condition_equal_mse=run_metric(
        validation, prediction(validation, None))['run_equal_mse'])}
    for name, features in feature_sets(horizon).items():
        score, model = fit_selected(train, validation, features, protocol['ridge_alphas'])
        models[name] = model
        selection[name] = dict(alpha=model['alpha'], validation_condition_equal_mse=score)
    support = float(train.time_min.max())
    scored = {}
    for role in SCORED_ROLES:
        part = rows[rows.run_id.isin(roles[role])]
        if part.empty:
            continue
        scored[role] = {}
        for name, model in models.items():
            yp = prediction(part, model)
            scored[role][name] = dict(run_metric(part, yp), time_support=c.time_support_metric(part, yp, support))
    return dict(horizon_min=horizon, max_train_origin_min=support, n_train_rows=int(len(train)),
                selection=selection, models=models, roles=scored)


def comparisons(scored):
    """Condition-equal RMSE comparisons and per-condition direction counts."""
    rmse = lambda name: scored[name]['run_equal_mse'] ** .5
    per_run = lambda name: scored[name]['per_run']
    history, clock = per_run('causal_history'), per_run('current_response_clock')
    persistence = per_run('persistence')
    return dict(
        rmse={name: rmse(name) for name in scored},
        history_vs_clock=dict(holds=rmse('causal_history') < rmse('current_response_clock'),
                              relative_reduction=1 - rmse('causal_history') / rmse('current_response_clock')),
        history_clock_vs_clock=dict(holds=rmse('history_clock') < rmse('current_response_clock'),
                                    relative_reduction=1 - rmse('history_clock') / rmse('current_response_clock')),
        n_conditions=len(history),
        conditions_history_beats_clock=sum(history[k]['mse'] < clock[k]['mse'] for k in history),
        conditions_history_beats_persistence=sum(history[k]['mse'] < persistence[k]['mse'] for k in history))


def hypotheses(analyses):
    by_horizon = {h: {role: comparisons(a['roles'][role]) for role in NEW_TEST_ROLES + ('viewed_reference',)
                      if role in a['roles']} for h, a in analyses.items()}
    primary = by_horizon['10']['test_new_protocol']
    return dict(H1_history_beats_clock_10min_new_protocol=primary['history_vs_clock']['holds'],
                H2_history_clock_beats_clock_10min_new_protocol=primary['history_clock_vs_clock']['holds'],
                by_horizon=by_horizon)


def leave_one_protocol_out(blocks, protocol, primary):
    roles = protocol['roles']
    eligible = [k for role in ('train', 'validation') + NEW_TEST_ROLES for k in roles[role] if k in blocks]
    groups = {}
    for key in eligible:
        name = key.rsplit('_', 1)[0]
        groups.setdefault('fgf_mixed_new' if name == 'fgf_mixed' else name, []).append(key)
    rows = c.forecasting_rows({k: blocks[k] for k in eligible}, 10.)
    features = feature_sets(10.)
    out = {}
    for group, held in groups.items():
        train, test = rows[~rows.run_id.isin(held)], rows[rows.run_id.isin(held)]
        out[group] = dict(conditions=held)
        for name in LOPO_MODELS:
            model = None if name == 'persistence' else fit_ridge(train, features[name], primary['selection'][name]['alpha'])
            out[group][name] = run_metric(test, prediction(test, model))['run_equal_mse'] ** .5
    return out


def sp60_timing_sensitivity(blocks, protocol, analyses):
    keys = [k for k in protocol['roles']['test_new_protocol_inferred_timing'] if k in blocks]
    out = {}
    for label, schedule in SP60_SHIFTS.items():
        shifted = {k: dict(blocks[k], protocol=schedule) for k in keys}
        out[label] = {}
        for h, analysis in analyses.items():
            rows = c.forecasting_rows(shifted, float(h))
            out[label][h] = {name: run_metric(rows, prediction(rows, model))['run_equal_mse'] ** .5
                             for name, model in analysis['models'].items()}
    return out


def main():
    protocol = load_protocol()
    blocks, inventory = load_blocks(protocol)
    analyses = {f'{h:g}': horizon_analysis(blocks, protocol, float(h)) for h in protocol['horizons_min']}
    result = dict(
        status='complete', protocol_sha256=protocol_sha256(),
        frozen_utc=json.loads(FREEZE.read_text(encoding='utf-8'))['frozen_utc'],
        inventory=inventory, mixed_normalization=mixed_normalization(protocol),
        horizons=analyses, hypotheses=hypotheses(analyses),
        leave_one_protocol_out=leave_one_protocol_out(blocks, protocol, analyses['10']),
        sp60_timing_sensitivity=sp60_timing_sensitivity(blocks, protocol, analyses),
        limitations=['Author-exported normalized FRET; conditions are not verified independent replicates.',
                     'New mixed concentrations share one experiment with the viewed reference.',
                     'The 60-min pulse timing is inferred from author simulations.',
                     'Prediction under recorded protocols only; no controller or counterfactual action is evaluated.'])
    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(result, indent=2) + '\n', newline='\n')
    summary = {h: {role: {k: round(v, 6) for k, v in comp['rmse'].items()}
                   for role, comp in by_role.items()} for h, by_role in result['hypotheses']['by_horizon'].items()}
    print(json.dumps(dict(H1=result['hypotheses']['H1_history_beats_clock_10min_new_protocol'],
                          H2=result['hypotheses']['H2_history_clock_beats_clock_10min_new_protocol'],
                          rmse=summary), indent=2))


if __name__ == '__main__':
    main()
