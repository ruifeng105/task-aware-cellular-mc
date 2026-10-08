"""E3: from response prediction to waiting decisions on B3.

Implements configs/b3_predict_decide_protocol.json, frozen before its only run. Six predictors (history mean,
current and smoothed kernels, ARX, NARX, B3 belief) predict the recovery statistic of Eq. (2) and the reporter
10 min ahead from the information available at each decision time. A monotone map fitted on out-of-fold
predictions of the reference receivers turns the predicted statistic into a success probability; the waiting
action thresholds its running maximum with the certified calibration of E1. Forecast error, window error,
probability calibration, false ready calls and completion are reported on the same test receivers.
Simulation of one published model structure, not evidence about measured cells.
Usage: python b3_predict_decide.py
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')  # one BLAS thread per process: no oversubscription in pools
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json

import numpy as np
from scipy import stats

import b3_readiness as br
import b3_waiting as bw
import b3_waiting_budget as bb
import b3_waiting_frontier as fr
import b3_waiting_strict as st
import strict_eval as se

PROTOCOL = br.ROOT / 'configs/b3_predict_decide_protocol.json'
FREEZE = br.OUT / 'b3_predict_decide_protocol_freeze.json'
RESULTS = br.OUT / 'predict_decide_results.json'
NOISE, KAPPA, ALPHA = .005, br.KAPPA, .025
PREDICTORS = ('history_mean', 'current', 'smoothed', 'arx', 'narx', 'b3_belief')
FOLDS, FOLD_SEED, LAMBDAS = 5, 20261015, (1e-6, 1e-4, 1e-2, 1., 100.)
LAGS, AHEAD, SMOOTH_TAU = 5, 5, 16.            # 5 lags of 2 min; 10 min ahead = 5 grid steps
LATE = br.GRID >= 60.
ORIGINS = br.GRID.size - AHEAD                 # forecast origins t = 0, 2, ..., 110 min


def protocol_sha256():
    return hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()


def load_protocol():
    if protocol_sha256() != json.loads(FREEZE.read_text(encoding='utf-8'))['protocol_sha256']:
        raise SystemExit('Protocol differs from its freeze record; refusing to run.')
    st.load_protocol()
    return json.loads(PROTOCOL.read_text(encoding='utf-8'))


def rho_table(tables):
    return tables['rise'].astype(float) / tables['naive_rise'].astype(float)[None, :, None]


def features(series, h):
    """Causal ARX features (n, W, 8) from noisy observations (n, W) of history h."""
    n, width = series.shape
    k = np.arange(width)
    lags = [series[:, np.maximum(k - j, 0)] for j in range(LAGS)]
    t = np.broadcast_to(br.GRID / 120., (n, width))
    long = np.full((n, width), float(h))
    return np.stack(lags + [t, long, long * t], axis=-1)


def poly2(x):
    i, j = np.triu_indices(x.shape[-1])
    return np.concatenate([x, x[..., i] * x[..., j]], axis=-1)


class Ridge:
    """Ridge regression on standardized features with an unpenalized intercept; penalty per sample."""

    def __init__(self, lam):
        self.lam = lam

    def fit(self, x, y):
        self.mu, sd = x.mean(0), x.std(0)
        self.sd = np.where(sd > 0, sd, 1.)
        z = (x - self.mu) / self.sd
        self.y0 = y.mean()
        self.beta = np.linalg.solve(z.T @ z / len(y) + self.lam * np.eye(z.shape[1]), z.T @ (y - self.y0) / len(y))
        return self

    def predict(self, x):
        return ((x - self.mu) / self.sd) @ self.beta + self.y0


class IsotonicMap:
    """Non-decreasing step map from predicted rho to success probability (pool adjacent violators)."""

    def __init__(self, x, y):
        x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
        self.x, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
        means = np.bincount(inverse, weights=y) / counts
        self.y = -bb.pava_nonincreasing(-means, counts.astype(float))

    def __call__(self, q):
        q = np.asarray(q, float)
        index = np.clip(np.searchsorted(self.x, np.nan_to_num(q, nan=-np.inf), side='right') - 1, 0, None)
        return np.where(np.isnan(q), np.nan, self.y[index])


def design(observed, plants, model, ahead=False):
    """Stacked features of both histories (H, n, W or ORIGINS, d)."""
    x = np.stack([features(observed[h, plants], h) for h in range(observed.shape[0])])
    x = x[:, :, :ORIGINS] if ahead else x
    return poly2(x) if model == 'narx' else x


def ridge_targets(tables, rho, plants, ahead):
    if ahead:
        return tables['history_fret'].astype(float)[:, plants, AHEAD:]
    return rho[:, plants, :]


def fit_core(tables, observed, rho, refs, lambdas):
    """Everything learned from reference receivers `refs`."""
    fret = tables['history_fret'].astype(float)
    core = dict(refs=refs, rho=rho[:, refs, :],
                current=bw.PooledReadiness(fret[:, refs], rho[:, refs], bw.likelihood_sd(NOISE)),
                smoothed=bw.PooledReadiness(bw.ema(fret[:, refs], SMOOTH_TAU), rho[:, refs],
                                            bw.smoothed_sd(NOISE, SMOOTH_TAU)))
    for model in ('arx', 'narx'):
        for ahead in (False, True):
            x = design(observed, refs, model, ahead)
            y = ridge_targets(tables, rho, refs, ahead)
            core[(model, ahead)] = Ridge(lambdas[f'{model}_{"forecast" if ahead else "rho"}']).fit(
                x.reshape(-1, x.shape[-1]), y.ravel())
    return core


def belief(observed, trajectories, values):
    """B3-likelihood weighted mean of `values` (m, W) for observations (n, W) under one history."""
    log_w = -np.cumsum((trajectories[None] - observed[:, None]) ** 2, axis=-1) / (2 * bw.likelihood_sd(NOISE) ** 2)
    log_w -= log_w.max(axis=1, keepdims=True)
    w = np.exp(log_w)
    return (w / w.sum(axis=1, keepdims=True) * values[None]).sum(1)


def predict_core(core, tables, observed, plants):
    """{predictor: (rho_hat (H, n, W), reporter forecast (H, n, ORIGINS))}."""
    fret = tables['history_fret'].astype(float)
    refs, obs = core['refs'], observed[:, plants]
    smooth = bw.ema(obs, SMOOTH_TAU)
    shape = obs.shape
    out = {'history_mean': (np.broadcast_to(core['rho'].mean(1)[:, None, :], shape).copy(),
                            np.broadcast_to(fret[:, refs, AHEAD:].mean(1)[:, None, :], shape[:2] + (ORIGINS,)).copy()),
           'current': (core['current'].probability(obs), obs[..., :ORIGINS].copy()),
           'smoothed': (core['smoothed'].probability(smooth), smooth[..., :ORIGINS].copy())}
    for model in ('arx', 'narx'):
        parts = []
        for ahead in (False, True):
            x = design(observed, plants, model, ahead)
            parts.append(core[(model, ahead)].predict(x.reshape(-1, x.shape[-1])).reshape(x.shape[:-1]))
        out[model] = tuple(parts)
    ahead_values = np.concatenate([fret[:, refs, AHEAD:], np.repeat(fret[:, refs, -1:], AHEAD, axis=-1)], axis=-1)
    b_rho = np.stack([belief(obs[h], fret[h, refs], core['rho'][h]) for h in range(shape[0])])
    b_fc = np.stack([belief(obs[h], fret[h, refs], ahead_values[h]) for h in range(shape[0])])[..., :ORIGINS]
    out['b3_belief'] = (b_rho, b_fc)
    return out


def folds(refs):
    order = np.random.default_rng(FOLD_SEED).permutation(len(refs))
    return [np.sort(refs[part]) for part in np.array_split(order, FOLDS)]


def choose_lambdas(tables, observed, rho, refs):
    parts, chosen = folds(refs), {}
    for model in ('arx', 'narx'):
        for ahead in (False, True):
            errors = []
            for lam in LAMBDAS:
                sse = 0.
                for held in parts:
                    train = np.setdiff1d(refs, held)
                    x, y = design(observed, train, model, ahead), ridge_targets(tables, rho, train, ahead)
                    fit = Ridge(lam).fit(x.reshape(-1, x.shape[-1]), y.ravel())
                    xv, yv = design(observed, held, model, ahead), ridge_targets(tables, rho, held, ahead)
                    sse += float(((fit.predict(xv.reshape(-1, xv.shape[-1])) - yv.ravel()) ** 2).sum())
                errors.append(sse)
            chosen[f'{model}_{"forecast" if ahead else "rho"}'] = LAMBDAS[int(np.argmin(errors))]
    return chosen


def fit_bundle(tables, observed, rho, success, refs):
    """Ridge parameters by cross-validation over `refs`, calibration maps from out-of-fold predictions, and the
    final predictors trained on all of `refs`. Only reference receivers' rho and outcomes are read."""
    lambdas = choose_lambdas(tables, observed, rho, refs)
    oof = {name: np.zeros((rho.shape[0], len(refs), rho.shape[2])) for name in PREDICTORS}
    position = {int(p): i for i, p in enumerate(refs)}
    for held in folds(refs):
        core = fit_core(tables, observed, rho, np.setdiff1d(refs, held), lambdas)
        predicted = predict_core(core, tables, observed, held)
        rows = [position[int(p)] for p in held]
        for name in PREDICTORS:
            oof[name][:, rows, :] = predicted[name][0]
    labels = np.asarray(success, float)[:, refs, :]
    maps = {name: [IsotonicMap(oof[name][h], labels[h]) for h in range(rho.shape[0])] for name in PREDICTORS}
    return dict(core=fit_core(tables, observed, rho, refs, lambdas), maps=maps, **{'lambda': lambdas})


def predict(bundle, tables, observed, plants):
    return predict_core(bundle['core'], tables, observed, plants)


def probability(bundle, predicted):
    return {name: np.stack([bundle['maps'][name][h](predicted[name][0][h]) for h in range(predicted[name][0].shape[0])])
            for name in PREDICTORS}


def threshold_options(p, success, plants):
    k = fr.crossing_index(p, fr.THRESHOLDS)
    options = []
    for i, c in enumerate(fr.THRESHOLDS):
        ok, completion = fr.scored(success, plants, k[i], br.DEADLINE_MIN)
        options.append((dict(threshold=float(c)), ok, completion))
    return options


def fixed_options(success, plants):
    out = []
    for j, w in enumerate(br.GRID):
        k = np.full((success.shape[0], len(plants)), j)
        ok, completion = fr.scored(success, plants, k, br.DEADLINE_MIN)
        out.append((dict(wait_min=float(w)), ok, completion))
    return out


def certified(options, h):
    return se.certified_choice([list(reversed(options))], h, br.Q, ALPHA)


def calibration_metrics(p, labels):
    p, labels = p.ravel(), labels.ravel().astype(float)
    edges = np.minimum((p * 10).astype(int), 9)
    ece = sum((edges == b).mean() * abs(p[edges == b].mean() - labels[edges == b].mean())
              for b in range(10) if (edges == b).any())
    return float(((p - labels) ** 2).mean()), float(ece)


def run_split(r, tables=None, success=None):
    tables = dict(np.load(br.TABLES)) if tables is None else tables
    success = br.success_table(tables) if success is None else success
    rho, fret = rho_table(tables), tables['history_fret'].astype(float)
    R, C, T = st.roles(r)
    observed = bw.noisy_observations(tables, NOISE, seed=st.noise_seed(r))
    bundle = fit_bundle(tables, observed, rho, success, R)
    on_c, on_t = predict(bundle, tables, observed, C), predict(bundle, tables, observed, T)
    p_c, p_t = probability(bundle, on_c), probability(bundle, on_t)
    fixed_choice = {hist: certified(fixed_options(success, C), h) for h, hist in enumerate(br.HISTORIES)}
    k_fixed = np.stack([np.full(len(T), int(fixed_choice[hist]['wait_min'] / 2.) if fixed_choice[hist]['certified']
                                else br.GRID.size - 1) for hist in br.HISTORIES])
    fixed_ok, fixed_completion = fr.scored(success, T, k_fixed, br.DEADLINE_MIN)
    out = dict(r=r, lambdas=bundle['lambda'],
               fixed=dict(choice=fixed_choice, rules=st.per_history(fixed_ok, fixed_completion, br.GRID[k_fixed])),
               predictors={})
    for name in PREDICTORS:
        options = threshold_options(p_c[name], success, C)
        choice = {hist: certified(options, h) for h, hist in enumerate(br.HISTORIES)}
        thresholds = [choice[hist].get('threshold') for hist in br.HISTORIES]
        peak = np.maximum.accumulate(p_t[name], axis=-1)
        k, crossed = [], []
        for h, c in enumerate(thresholds):
            if c is None:
                k.append(np.full(len(T), br.GRID.size - 1))
                crossed.append(np.zeros(len(T), bool))
            else:
                hit = peak[h] >= c
                k.append(np.where(hit.any(-1), hit.argmax(-1), br.GRID.size - 1))
                crossed.append(hit.any(-1))
        k, crossed = np.stack(k), np.stack(crossed)
        ok, completion = fr.scored(success, T, k, br.DEADLINE_MIN)
        brier, ece = calibration_metrics(p_t[name], success[:, T, :])
        rho_hat, forecast = on_t[name]
        diff = (completion - fixed_completion).mean(0)
        out['predictors'][name] = dict(
            choice=choice,
            metrics=dict(forecast_rmse=float(np.sqrt(((forecast - fret[:, T, AHEAD:]) ** 2).mean())),
                         rho_rmse_all=float(np.sqrt(((rho_hat - rho[:, T, :]) ** 2).mean())),
                         rho_rmse_late=float(np.sqrt(((rho_hat - rho[:, T, :])[..., LATE] ** 2).mean())),
                         brier=brier, ece=ece),
            false_ready={hist: dict(probes_at_crossing=int(crossed[h].sum()), fallback_probes=int((~crossed[h]).sum()),
                                    failed_at_crossing=int((~ok[h] & crossed[h]).sum()),
                                    rate=float((~ok[h] & crossed[h]).sum() / crossed[h].sum()) if crossed[h].any() else None)
                         for h, hist in enumerate(br.HISTORIES)},
            rules=st.per_history(ok, completion, br.GRID[k]),
            paired=dict(completion=float(diff.mean()), completion_ci=se.paired_bootstrap(diff, st.BOOT, st.BOOT_SEED)))
    return out


def rate_stats(rates):
    known = [v for v in rates if v is not None]
    return dict(st.stats(known), splits=len(known)) if known else dict(median=None, min=None, max=None, splits=0)


def summarize(repeats):
    rows = [repeats[str(r)] for r in range(st.SPLITS)]
    metrics = ('forecast_rmse', 'rho_rmse_all', 'rho_rmse_late', 'brier', 'ece')
    spearman = {'forecast': [], 'window': []}
    for row in rows:
        completion = [row['predictors'][n]['rules']['all']['mean_completion_min'] for n in PREDICTORS]
        spearman['forecast'].append(float(stats.spearmanr([row['predictors'][n]['metrics']['forecast_rmse']
                                                           for n in PREDICTORS], completion).statistic))
        spearman['window'].append(float(stats.spearmanr([row['predictors'][n]['metrics']['rho_rmse_all']
                                                         for n in PREDICTORS], completion).statistic))
    return dict(
        metrics={n: {m: st.stats([row['predictors'][n]['metrics'][m] for row in rows]) for m in metrics}
                 for n in PREDICTORS},
        completion=dict({n: st.stats([row['predictors'][n]['rules']['all']['mean_completion_min'] for row in rows])
                         for n in PREDICTORS},
                        fixed=st.stats([row['fixed']['rules']['all']['mean_completion_min'] for row in rows])),
        paired_completion={n: st.stats([row['predictors'][n]['paired']['completion'] for row in rows]) for n in PREDICTORS},
        false_ready={n: {h: rate_stats([row['predictors'][n]['false_ready'][h]['rate'] for row in rows])
                         for h in br.HISTORIES} for n in PREDICTORS},
        meets_per_history={n: int(sum(all(row['predictors'][n]['rules'][h]['success_rate'] >= br.Q
                                          for h in br.HISTORIES) for row in rows)) for n in PREDICTORS},
        certified_both={n: int(sum(all(row['predictors'][n]['choice'][h]['certified'] for h in br.HISTORIES)
                                   for row in rows)) for n in PREDICTORS},
        spearman={k: dict(values=v, **st.stats(v)) for k, v in spearman.items()})


def statements(repeats, summary):
    primary = repeats['0']['predictors']
    order = lambda key: sorted(PREDICTORS, key=key)
    meeting = [n for n in PREDICTORS if all(primary[n]['rules'][h]['success_rate'] >= br.Q for h in br.HISTORIES)]
    best_forecast = order(lambda n: primary[n]['metrics']['forecast_rmse'])[0]
    fastest = min(meeting, key=lambda n: primary[n]['rules']['all']['mean_completion_min']) if meeting else None
    focus = ('arx', 'narx', 'b3_belief', 'smoothed')
    return {
        'E3-S1': dict(by_forecast_rmse=order(lambda n: primary[n]['metrics']['forecast_rmse']),
                      by_window_rmse_late=order(lambda n: primary[n]['metrics']['rho_rmse_late']),
                      by_brier=order(lambda n: primary[n]['metrics']['brier']),
                      by_completion=order(lambda n: primary[n]['rules']['all']['mean_completion_min']),
                      meeting_per_history=meeting, lowest_forecast_error=best_forecast, fastest_meeting=fastest,
                      lowest_forecast_is_fastest_meeting=best_forecast == fastest),
        'E3-S2': dict(forecast_vs_completion_median=summary['spearman']['forecast']['median'],
                      window_vs_completion_median=summary['spearman']['window']['median']),
        'E3-S3': {n: dict(split0=dict(primary[n]['metrics'], completion=primary[n]['rules']['all']['mean_completion_min'],
                                      paired_completion=primary[n]['paired']['completion'],
                                      false_ready={h: primary[n]['false_ready'][h]['rate'] for h in br.HISTORIES}),
                          medians=dict({m: v['median'] for m, v in summary['metrics'][n].items()},
                                       completion=summary['completion'][n]['median'],
                                       paired_completion=summary['paired_completion'][n]['median']))
                  for n in focus}}


def _job(r):
    return run_split(r)


def main():
    load_protocol()
    with ProcessPoolExecutor(max_workers=min(20, os.cpu_count() or 1)) as pool:
        results = list(pool.map(_job, range(st.SPLITS)))
    repeats = {str(r): result for r, result in zip(range(st.SPLITS), results)}
    summary = summarize(repeats)
    result = dict(status='complete', protocol_sha256=protocol_sha256(),
                  frozen_utc=json.loads(FREEZE.read_text(encoding='utf-8'))['frozen_utc'],
                  repeats=repeats, summary=summary, statements=statements(repeats, summary),
                  limitations=['Simulation of one published model structure in which B3 is the plant.',
                               'ARX and NARX are not structurally informed; the kernels and B3 belief use model trajectories.',
                               'No counterfactual decision evaluation on measured cells.'])
    RESULTS.write_text(json.dumps(result, indent=2) + '\n', newline='\n')
    for n in PREDICTORS:
        m = summary['metrics'][n]
        print(n, {k: round(v['median'], 4) for k, v in m.items()}, 'T', round(summary['completion'][n]['median'], 2),
              'dT', round(summary['paired_completion'][n]['median'], 2), 'meets', summary['meets_per_history'][n],
              'cert', summary['certified_both'][n], 'FR', {h: v['median'] for h, v in summary['false_ready'][n].items()})
    print('fixed T', summary['completion']['fixed'])
    print(json.dumps(result['statements']['E3-S1'], indent=1), result['statements']['E3-S2'])


if __name__ == '__main__':
    main()
