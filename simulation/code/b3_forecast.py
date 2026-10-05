"""B3 matched-feedback forecaster and the nested M0-M3 comparison.

Implements configs/b3_forecast_protocol.json, a post hoc reference analysis on
already inspected roles. Refuses to run if the protocol differs from its
freeze record or the B3 acceptance check has not passed. The posterior
samples act as a fixed particle set weighted by each cell's past reporter.
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import b3_model as b3
import calibrate_fgf2 as c
import expanded_fgf2 as e
from history_baselines import predict, run_metric

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'configs/b3_forecast_protocol.json'
OUT = ROOT / 'results/b3'
FREEZE = OUT / 'b3_forecast_protocol_freeze.json'
RESULTS = OUT / 'forecast_results.json'
KEYS = ['run_id', 'cell_id', 'time_min']
FIXED_FORECASTS = ('b3_open_loop', 'b3_anchored', 'M2_b3_feedback')
RIDGE_FORECASTS = ('M0_current_clock', 'M1_causal_history', 'M3_history_plus_b3')


def protocol_sha256():
    return hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()


def load_protocol():
    if protocol_sha256() != json.loads(FREEZE.read_text(encoding='utf-8'))['protocol_sha256']:
        raise SystemExit('Protocol differs from its freeze record; refusing to run.')
    if json.loads((OUT / 'verification_results.json').read_text(encoding='utf-8'))['status'] != 'passed':
        raise SystemExit('B3 acceptance check has not passed.')
    return json.loads(PROTOCOL.read_text(encoding='utf-8'))


def label(sigma):
    return sigma if isinstance(sigma, str) else f'{sigma:g}'


def simulate_population(blocks, workers=None):
    """Noise-free FRET of every posterior sample on a 1-min grid per condition."""
    posterior = b3.load_posterior()
    keys = list(blocks)
    grids = {k: tuple(np.arange(0., float(blocks[k]['time'].max()) + 1., 1.)) for k in keys}
    jobs = [(tuple(row), blocks[k]['protocol'], blocks[k]['concentration'], grids[k]) for k in keys for row in posterior]
    with ProcessPoolExecutor(max_workers=workers or min(32, os.cpu_count() or 1)) as pool:
        out = list(pool.map(b3._simulate_job, jobs, chunksize=64))
    n = len(posterior)
    return {k: np.array(out[i * n:(i + 1) * n]) for i, k in enumerate(keys)}


def feedback_weights(observed, trajectories, sigma):
    """Column i weights the samples by the observations up to index i only."""
    samples, length = trajectories.shape
    if sigma == 'equal_weights':
        return np.full((samples, length), 1. / samples)
    log_weights = -np.cumsum((trajectories - observed[None, :]) ** 2, axis=1) / (2 * sigma ** 2)
    log_weights -= log_weights.max(axis=0, keepdims=True)
    weights = np.exp(log_weights)
    return weights / weights.sum(axis=0, keepdims=True)


def b3_columns(blocks, population, horizon, sigmas):
    """B3 forecasts on exactly the rows that calibrate_fgf2.forecasting_rows keeps."""
    lag, records = int(horizon / 2), []
    for key, block in blocks.items():
        t = block['time']
        index = np.rint(t).astype(int)
        observed_f = population[key][:, index]
        # Clipped indices only occur after the last eligible forecast origin.
        future = np.minimum(index + int(horizon), population[key].shape[1] - 1)
        increment = population[key][:, future] - observed_f
        open_loop = (observed_f + increment).mean(0)
        mean_increment = increment.mean(0)
        eligible = [i for i, now in enumerate(t) if i >= 1 and now - t[0] >= 10 and i + lag < len(t)]
        for cell, y in enumerate(block['y']):
            deltas = {label(s): (feedback_weights(y, observed_f, s) * increment).sum(0) for s in sigmas}
            for i in eligible:
                record = dict(run_id=key, cell_id=str(cell), time_min=float(t[i]),
                              b3_open_loop=open_loop[i], b3_anchored=y[i] + mean_increment[i])
                record.update({f'b3_delta_{s}': d[i] for s, d in deltas.items()})
                records.append(record)
    return pd.DataFrame(records)


def forecast_rows(blocks, population, horizon, sigmas):
    rows = c.forecasting_rows(blocks, horizon)
    merged = rows.merge(b3_columns(blocks, population, horizon, sigmas), on=KEYS, how='left', validate='one_to_one')
    assert len(merged) == len(rows) and not merged.filter(like='b3_').isna().any().any()
    return merged


def prediction(rows, name, models, sigma):
    if name in ('b3_open_loop', 'b3_anchored'):
        return rows[name].to_numpy()
    if name == 'M2_b3_feedback':
        return rows.current_y.to_numpy() + rows[f'b3_delta_{sigma}'].to_numpy()
    return predict(rows.assign(b3_delta=rows[f'b3_delta_{sigma}']), models[name]['model'])


def horizon_analysis(rows, protocol, roles, horizon):
    validation = rows[rows.run_id.isin(roles['validation'])]
    sigma_scores = {label(s): run_metric(validation, validation.current_y.to_numpy()
                                         + validation[f'b3_delta_{label(s)}'].to_numpy())['run_equal_mse']
                    for s in protocol['sigma_grid']}
    sigma = min(sigma_scores, key=sigma_scores.get)
    data = rows.assign(b3_delta=rows[f'b3_delta_{sigma}'])
    train, validation = data[data.run_id.isin(roles['train'])], data[data.run_id.isin(roles['validation'])]
    features = e.feature_sets(horizon)
    columns = {'M0_current_clock': features['current_response_clock'],
               'M1_causal_history': features['causal_history'],
               'M3_history_plus_b3': features['causal_history'] + ['b3_delta']}
    models = {}
    for name, cols in columns.items():
        score, model = e.fit_selected(train, validation, cols, protocol['ridge_alphas'])
        models[name] = dict(alpha=model['alpha'], validation_condition_equal_mse=score, model=model)
    return dict(sigma=sigma, sigma_validation_mse=sigma_scores, models=models)


def score(rows, selection, roles):
    scores = {}
    for role in e.SCORED_ROLES:
        part = rows[rows.run_id.isin(roles[role])]
        if not part.empty:
            scores[role] = {name: run_metric(part, prediction(part, name, selection['models'], selection['sigma']))
                            for name in FIXED_FORECASTS + RIDGE_FORECASTS}
    return scores


def comparisons(scores):
    out = {}
    for horizon, roles in scores.items():
        out[horizon] = {}
        for role, models in roles.items():
            rmse = {name: m['run_equal_mse'] ** .5 for name, m in models.items()}
            per_run = {name: m['per_run'] for name, m in models.items()}
            wins = lambda a, b: sum(per_run[a][k]['mse'] < per_run[b][k]['mse'] for k in per_run[a])
            out[horizon][role] = dict(rmse=rmse, n_conditions=len(per_run['M2_b3_feedback']),
                                      M3_beats_M2=dict(holds=rmse['M3_history_plus_b3'] < rmse['M2_b3_feedback'],
                                                       conditions=wins('M3_history_plus_b3', 'M2_b3_feedback')),
                                      M2_beats_M0=dict(holds=rmse['M2_b3_feedback'] < rmse['M0_current_clock'],
                                                       conditions=wins('M2_b3_feedback', 'M0_current_clock')),
                                      M2_beats_M1=dict(holds=rmse['M2_b3_feedback'] < rmse['M1_causal_history'],
                                                       conditions=wins('M2_b3_feedback', 'M1_causal_history')))
    primary = out['10']['test_new_protocol']
    return dict(C1_primary_M3_beats_M2=primary['M3_beats_M2']['holds'], C2_M2_beats_M0=primary['M2_beats_M0']['holds'],
                C3_M2_beats_M1=primary['M2_beats_M1']['holds'], by_horizon=out)


def load_inputs():
    protocol = load_protocol()
    expanded_protocol = e.load_protocol()
    blocks, _ = e.load_blocks(expanded_protocol)
    return protocol, expanded_protocol['roles'], blocks, simulate_population(blocks)


def score_saved(saved):
    """Recompute role scores from saved sigma and ridge models, without retuning."""
    protocol, roles, blocks, population = load_inputs()
    out = {}
    for horizon, selection in saved['selection'].items():
        sigma = selection['sigma'] if selection['sigma'] == 'equal_weights' else float(selection['sigma'])
        rows = forecast_rows(blocks, population, float(horizon), [sigma])
        out[horizon] ={role: {name: metric['run_equal_mse'] for name, metric in models.items()}
                        for role, models in score(rows, selection, roles).items()}
    return out


def main():
    protocol, roles, blocks, population = load_inputs()
    expanded = json.loads(e.RESULTS.read_text(encoding='utf-8'))
    selection, scores, consistency = {}, {}, {}
    for horizon in protocol['horizons_min']:
        h = f'{horizon:g}'
        rows = forecast_rows(blocks, population, float(horizon), protocol['sigma_grid'])
        selection[h] = horizon_analysis(rows, protocol, roles, float(horizon))
        scores[h] = score(rows, selection[h], roles)
        consistency[h] = max(abs(scores[h][role][ours]['run_equal_mse'] - expanded['horizons'][h]['roles'][role][theirs]['run_equal_mse'])
                             for role in scores[h] for ours, theirs in (('M0_current_clock', 'current_response_clock'),
                                                                        ('M1_causal_history', 'causal_history')))
    result = dict(status='complete', protocol_sha256=protocol_sha256(),
                  frozen_utc=json.loads(FREEZE.read_text(encoding='utf-8'))['frozen_utc'],
                  n_posterior_samples=len(b3.load_posterior()), selection=selection, scores=scores,
                  comparisons=comparisons(scores),
                  max_abs_difference_M0_M1_vs_expanded_results=consistency,
                  limitations=['Post hoc reference analysis on inspected roles; only sigma and alpha were tuned, on validation.',
                               'Posterior samples express population-mean uncertainty, not cell heterogeneity.',
                               'B3 architecture selection used mixed and single 5-min conditions at 2.5/250 ng/ml.',
                               'Conditions are not verified independent replicates.'])
    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(result, indent=2) + '\n', newline='\n')
    print(json.dumps({h: {role: {k: round(v, 5) for k, v in comp['rmse'].items()} for role, comp in roles_.items()}
                      for h, roles_ in result['comparisons']['by_horizon'].items()}, indent=2))
    print(json.dumps({k: v for k, v in result['comparisons'].items() if k != 'by_horizon'}),
          'sigma:', {h: s['sigma'] for h, s in selection.items()}, 'consistency:', consistency)


if __name__ == '__main__':
    main()
