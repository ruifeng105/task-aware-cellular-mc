"""Covariance-aware likelihood for the history estimator under AR(1) observation noise on B3.

Implements configs/b3_ar1_likelihood_protocol.json (review item P1), frozen
before its only run. Under AR(1) observation noise it separates what
recalibrating the thresholds recovers from what modelling the correlation
adds: frozen white-noise rules, rules recalibrated on AR(1) calibration data
with the independent-noise likelihood, and the history estimator with an
AR(1) likelihood recalibrated on the same data. The frozen arm of split r
uses the white-noise choices of the robustness study on that split (version
3 for r = 0). Simulation of one published model structure.
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os

import numpy as np

import b3_readiness as br
import b3_waiting as bw
import b3_waiting_baselines as bl
import b3_waiting_robustness as rb
import b3_waiting_stress as st

PROTOCOL = br.ROOT / 'configs/b3_ar1_likelihood_protocol.json'
FREEZE = br.OUT / 'b3_ar1_likelihood_protocol_freeze.json'
RESULTS = br.OUT / 'ar1_likelihood_results.json'
TRUE_SD, TRUE_RHO = .005, st.RHO
PAIRS = (('ar1_history', 'independent_history'), ('ar1_history', 'frozen_history'))


def protocol_sha256():
    return hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()


def load_protocol():
    if protocol_sha256() != json.loads(FREEZE.read_text(encoding='utf-8'))['protocol_sha256']:
        raise SystemExit('Protocol differs from its freeze record; refusing to run.')
    st.load_protocol()
    return json.loads(PROTOCOL.read_text(encoding='utf-8'))


def ar1_observations(tables, seed, sd=TRUE_SD, rho=TRUE_RHO):
    """b3_waiting_stress.ar1_observations with the seed as a parameter."""
    fret = tables['history_fret'].astype(float)
    noise = np.empty_like(fret)
    scale = np.sqrt(1 - rho ** 2) * sd
    for h in range(fret.shape[0]):
        for plant in range(fret.shape[1]):
            eps = np.random.default_rng(seed + 2 * plant + h).normal(0., 1., fret.shape[2])
            noise[h, plant, 0] = sd * eps[0]
            for k in range(1, fret.shape[2]):
                noise[h, plant, k] = rho * noise[h, plant, k - 1] + scale * eps[k]
    return fret + noise


def estimate_noise(tables, observed, cal):
    """Pooled s.d. and lag-1 autocorrelation of the calibration residuals."""
    residual = observed[:, cal, :] - tables['history_fret'][:, cal, :].astype(float)
    lagged = np.corrcoef(residual[..., 1:].ravel(), residual[..., :-1].ravel())[0, 1]
    return float(residual.std()), float(lagged)


def ar1_log_weights(observed, trajectories, sigma, rho):
    """Cumulative AR(1) Gaussian log-likelihood of each reference trajectory, (references, frames)."""
    e = trajectories - observed[None, :]
    innovation = np.empty_like(e)
    innovation[:, 0] = e[:, 0]
    innovation[:, 1:] = (e[:, 1:] - rho * e[:, :-1]) / np.sqrt(1 - rho ** 2)
    return -np.cumsum(innovation ** 2, axis=1) / (2 * sigma ** 2)


def ar1_history_probability(observed, trajectories, success, sigma, rho):
    log_w = ar1_log_weights(observed, trajectories, sigma, rho)
    log_w -= log_w.max(axis=0, keepdims=True)
    weights = np.exp(log_w)
    return (weights / weights.sum(axis=0, keepdims=True) * success).sum(0)


def ar1_probabilities(tables, success, observed, sigma, rho, reference, plants):
    fret = tables['history_fret'].astype(float)
    out = np.zeros((fret.shape[0], len(plants), fret.shape[2]))
    for h in range(fret.shape[0]):
        for i, p in enumerate(plants):
            keep = reference[reference != p]
            out[h, i] = ar1_history_probability(observed[h, p], fret[h, keep], success[h, keep], sigma, rho)
    return out


def median_ess(tables, observed, sigma, rho, reference, plants):
    """Median effective sample size of the history weights at 90 min after the 30-min command."""
    fret, k = tables['history_fret'][1].astype(float), int(np.flatnonzero(br.GRID == st.ESS_MIN)[0])
    values = []
    for p in plants:
        log_w = ar1_log_weights(observed[1, p], fret[reference], sigma, rho)[:, k]
        w = np.exp(log_w - log_w.max())
        values.append(w.sum() ** 2 / (w ** 2).sum())
    return float(np.median(values))


def seeds(r):
    return (bw.SPLIT_SEED, st.AR_SEED) if r == 0 else (bw.SPLIT_SEED + r, st.AR_SEED + 10000 * r)


def run_repeat(r):
    tables = dict(np.load(br.TABLES))
    success = br.success_table(tables)
    split_seed, ar_seed = seeds(r)
    cal, ev = bw.split_samples(success.shape[1], seed=split_seed)
    observed = ar1_observations(tables, ar_seed)
    sigma, rho = estimate_noise(tables, observed, cal)
    frozen_choices = json.loads(rb.RESULTS.read_text(encoding='utf-8'))['repeats'][str(r)]['0.005']['choices']
    frozen_p = bw.estimator_probabilities(tables, success, .005, cal, ev, observed)
    independent_choices = bw.calibration_choices(tables, sigma, (cal, ev), observed)
    independent_p = bw.estimator_probabilities(tables, success, sigma, cal, ev, observed)
    outcomes, choices = {}, {'frozen': frozen_choices, 'independent': independent_choices}
    for arm, chosen, p in (('frozen', frozen_choices, frozen_p), ('independent', independent_choices, independent_p)):
        for name in ('current', 'history'):
            outcomes[f'{arm}_{name}'] = rb.plant_outcomes(p[name], success, ev, chosen[name]['threshold'])
        outcomes[f'{arm}_smoothed'] = rb.plant_outcomes(p[f"smoothed_{chosen['smoothed']['tau_min']:g}"], success, ev,
                                                        chosen['smoothed']['threshold'])
    for arm, (s, q) in (('ar1', (sigma, rho)), ('ar1_true', (TRUE_SD, TRUE_RHO))):
        threshold, calibration, reached = bw.calibrate_threshold(ar1_probabilities(tables, success, observed, s, q, cal, cal),
                                                                 success, cal)
        choices[arm] = dict(sigma=s, rho=q, threshold=threshold, target_reached=reached, calibration=calibration)
        outcomes[f'{arm}_history'] = rb.plant_outcomes(ar1_probabilities(tables, success, observed, s, q, cal, ev), success, ev, threshold)
    outcomes['fixed'] = rb.fixed_plant_outcomes(success, ev, independent_choices['fixed']['wait_min'])
    conditioned = bl.conditioned_fixed(success, cal)
    choices['conditioned'] = dict(waits_min=conditioned['waits_min'], target_reached=conditioned['target_reached'])
    outcomes['conditioned'] = bl.plant_fixed(success, ev, conditioned['waits_min'])
    per_sample = {name: (ok.mean(0), completion.mean(0)) for name, (ok, completion) in outcomes.items()}
    differences = {f'{a}-{b}': dict(success=per_sample[a][0] - per_sample[b][0], completion=per_sample[a][1] - per_sample[b][1])
                   for a, b in PAIRS}
    result = dict(r=r, split_seed=split_seed, ar_seed=ar_seed, estimated=dict(sigma=sigma, rho=rho),
                  choices=json.loads(json.dumps(choices)), rules={name: rb.summarise(*pair) for name, pair in outcomes.items()},
                  paired={pair: {key: float(v.mean()) for key, v in d.items()} for pair, d in differences.items()},
                  median_ess_90min=dict(independent=median_ess(tables, observed, max(sigma, bw.LIKELIHOOD_FLOOR), 0., cal, ev),
                                        ar1=median_ess(tables, observed, sigma, rho, cal, ev)))
    if r == 0:
        result['bootstrap'] = rb.bootstrap(differences)
    return result


def summarize(repeats):
    rows = [repeats[str(r)] for r in range(rb.REPEATS)]
    names = list(rows[0]['rules'])
    return dict(reached_on_evaluation={n: int(sum(row['rules'][n]['all']['success_rate'] >= br.Q for row in rows)) for n in names},
                success={n: rb.stats([row['rules'][n]['all']['success_rate'] for row in rows]) for n in names},
                completion={n: rb.stats([row['rules'][n]['all']['mean_completion_min'] for row in rows]) for n in names},
                paired={pair: {k: rb.stats([row['paired'][pair][k] for row in rows]) for k in ('success', 'completion')}
                        for pair in rows[0]['paired']},
                estimated={k: rb.stats([row['estimated'][k] for row in rows]) for k in ('sigma', 'rho')},
                median_ess_90min={k: rb.stats([row['median_ess_90min'][k] for row in rows]) for k in ('independent', 'ar1')},
                ar1_more_reliable_than_independent=int(sum(row['paired']['ar1_history-independent_history']['success'] > 0 for row in rows)))


def main():
    load_protocol()
    with ProcessPoolExecutor(max_workers=min(rb.REPEATS, os.cpu_count() or 1)) as pool:
        results = list(pool.map(run_repeat, range(rb.REPEATS)))
    repeats = {str(row['r']): row for row in results}
    result = dict(status='complete', protocol_sha256=protocol_sha256(),
                  frozen_utc=json.loads(FREEZE.read_text(encoding='utf-8'))['frozen_utc'],
                  repeats=repeats, summary=summarize(repeats),
                  limitations=['Simulation of one published model structure; not evidence about measured cells.',
                               'The AR(1) parameters are estimated from calibration residuals, which a real experiment would have to measure.',
                               'Mismatch changes only the observation model; dynamics and task are unchanged.'])
    RESULTS.write_text(json.dumps(result, indent=2) + '\n', newline='\n')
    r0 = repeats['0']
    print('estimated', r0['estimated'], 'ess', r0['median_ess_90min'])
    print('r0', {n: (round(100 * v['all']['success_rate'], 1), round(v['all']['mean_completion_min'], 1)) for n, v in r0['rules'].items()})
    print('r0 paired', r0['paired'], 'boot', r0['bootstrap'])
    s = result['summary']
    print('reached', s['reached_on_evaluation'])
    print('success med', {n: round(100 * v['median'], 1) for n, v in s['success'].items()})
    print('completion med', {n: round(v['median'], 1) for n, v in s['completion'].items()})
    print('paired', s['paired'], 'ess', s['median_ess_90min'], 'ar1 more reliable', s['ar1_more_reliable_than_independent'])


if __name__ == '__main__':
    main()
