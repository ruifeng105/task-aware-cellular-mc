"""Calibrated waiting rules on B3: threshold calibration, smoothed baseline, noise ablation.

Implements configs/b3_waiting_protocol.json (readiness protocol version 3),
frozen before any version-3 outcome. Reuses the version-2 simulation tables
and task. Posterior samples are split once: estimators, thresholds and the
comparator fixed wait use calibration samples only; success and completion
time are reported on evaluation samples. Simulation of one published model
structure, not evidence about measured cells.
"""
import hashlib
import json

import numpy as np

import b3_readiness as br

PROTOCOL = br.ROOT / 'configs/b3_waiting_protocol.json'
FREEZE = br.OUT / 'b3_waiting_protocol_freeze.json'
RESULTS = br.OUT / 'waiting_results.json'
SPLIT_SEED = 20261006
NOISE_LEVELS = (0., .005)
LIKELIHOOD_FLOOR = .001
THRESHOLDS = (.5, .6, .7, .8, .85, .9, .95, .975, .99)
SMOOTH_TAUS = (2., 4., 8., 16.)


def protocol_sha256():
    return hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()


def load_protocol():
    if protocol_sha256() != json.loads(FREEZE.read_text(encoding='utf-8'))['protocol_sha256']:
        raise SystemExit('Protocol differs from its freeze record; refusing to run.')
    return json.loads(PROTOCOL.read_text(encoding='utf-8'))


def split_samples(n, seed=SPLIT_SEED):
    order = np.random.default_rng(seed).permutation(n)
    return np.sort(order[:n // 2]), np.sort(order[n // 2:])


def noisy_observations(tables, noise, seed=None):
    """Version-2 draws by default; with `seed`, the same per-plant scheme (seed + 2 x plant + history)."""
    fret = tables['history_fret'].astype(float)
    if seed is None:
        return fret + (noise / br.NOISE_SD) * (br.observations(tables) - fret)
    draws = np.empty_like(fret)
    for h in range(fret.shape[0]):
        for plant in range(fret.shape[1]):
            draws[h, plant] = np.random.default_rng(seed + 2 * plant + h).normal(0., 1., fret.shape[2])
    return fret + noise * draws


def ema(series, tau):
    a = np.exp(-2. / tau)
    out = np.empty_like(series, dtype=float)
    out[..., 0] = series[..., 0]
    for k in range(1, series.shape[-1]):
        out[..., k] = a * out[..., k - 1] + (1 - a) * series[..., k]
    return out


def likelihood_sd(noise):
    return max(noise, LIKELIHOOD_FLOOR)


def smoothed_sd(noise, tau):
    a = np.exp(-2. / tau)
    return max(noise * np.sqrt((1 - a) / (1 + a)), LIKELIHOOD_FLOOR)


class PooledReadiness:
    """P(success | value) by a Gaussian kernel over pooled (state, success) pairs; arrays are (histories, samples, waits)."""

    def __init__(self, states, success, sigma, points=4001):
        self.states, self.success, self.sigma = states.astype(float), success.astype(float), sigma
        f, s = self.states.ravel(), self.success.ravel()
        self.grid = np.linspace(f.min() - 6 * sigma, f.max() + 6 * sigma, points)
        self.s0, self.s1 = np.zeros(points), np.zeros(points)
        for chunk in np.array_split(np.arange(points), 80):
            kernel = np.exp(-(self.grid[chunk, None] - f[None, :]) ** 2 / (2 * sigma ** 2))
            self.s0[chunk], self.s1[chunk] = kernel.sum(1), kernel @ s

    def probability(self, observed, exclude=None):
        s0, s1 = np.interp(observed, self.grid, self.s0), np.interp(observed, self.grid, self.s1)
        if exclude is not None:
            own_f, own_s = self.states[:, exclude, :].ravel(), self.success[:, exclude, :].ravel()
            own = np.exp(-(observed[:, None] - own_f[None, :]) ** 2 / (2 * self.sigma ** 2))
            s0, s1 = s0 - own.sum(1), s1 - own @ own_s
        return np.where(s0 > 1e-9, s1 / np.maximum(s0, 1e-300), 0.)


def reference_history_probability(observed, trajectories, success, sigma):
    log_w = -np.cumsum((trajectories - observed[None, :]) ** 2, axis=1) / (2 * sigma ** 2)
    log_w -= log_w.max(axis=0, keepdims=True)
    weights = np.exp(log_w)
    return (weights / weights.sum(axis=0, keepdims=True) * success).sum(0)


def estimator_probabilities(tables, success, noise, reference, plants, observed=None):
    """Success-probability series per estimator, arrays (histories, len(plants), waits); a plant inside
    `reference` is left out of its own reference set."""
    fret = tables['history_fret'].astype(float)
    observed = noisy_observations(tables, noise) if observed is None else observed
    position = {int(p): i for i, p in enumerate(reference)}
    estimators = {'current': (PooledReadiness(fret[:, reference], success[:, reference], likelihood_sd(noise)), observed)}
    for tau in SMOOTH_TAUS:
        estimators[f'smoothed_{tau:g}'] = (PooledReadiness(ema(fret[:, reference], tau), success[:, reference],
                                                           smoothed_sd(noise, tau)), ema(observed, tau))
    out = {name: np.zeros((fret.shape[0], len(plants), fret.shape[2])) for name in list(estimators) + ['history']}
    for h in range(fret.shape[0]):
        for i, p in enumerate(plants):
            for name, (model, series) in estimators.items():
                out[name][h, i] = model.probability(series[h, p], exclude=position.get(int(p)))
            keep = reference[reference != p]
            out['history'][h, i] = reference_history_probability(observed[h, p], fret[h, keep], success[h, keep],
                                                                 likelihood_sd(noise))
    return out


def outcomes(probability, success, plants, threshold):
    k = np.apply_along_axis(br.first_crossing, -1, probability, threshold)
    ok = np.take_along_axis(success[:, plants, :], k[..., None], -1)[..., 0]
    completion = np.where(ok, br.GRID[k] + br.WINDOW_MIN, br.DEADLINE_MIN)
    return dict(success_rate=float(ok.mean()), mean_completion_min=float(completion.mean()),
                mean_wait_min=float(br.GRID[k].mean()))


def calibrate_threshold(probability, success, plants):
    for threshold in THRESHOLDS:
        outcome = outcomes(probability, success, plants, threshold)
        if outcome['success_rate'] >= br.Q:
            return threshold, outcome, True
    return THRESHOLDS[-1], outcomes(probability, success, plants, THRESHOLDS[-1]), False


def fixed_outcome(success, plants, wait):
    k = int(np.flatnonzero(br.GRID == wait)[0])
    ok = success[:, plants, k]
    return dict(success_rate=float(ok.mean()),
                mean_completion_min=float(np.where(ok, wait + br.WINDOW_MIN, br.DEADLINE_MIN).mean()),
                mean_wait_min=float(wait))


def calibration_choices(tables, noise, split=None, observed=None):
    success = br.success_table(tables)
    cal, _ = split_samples(success.shape[1]) if split is None else split
    probability = estimator_probabilities(tables, success, noise, cal, cal, observed)
    choices = {}
    for name in ('current', 'history'):
        threshold, outcome, reached = calibrate_threshold(probability[name], success, cal)
        choices[name] = dict(threshold=threshold, target_reached=reached, calibration=outcome)
    smoothed = []
    for tau in SMOOTH_TAUS:
        threshold, outcome, reached = calibrate_threshold(probability[f'smoothed_{tau:g}'], success, cal)
        smoothed.append(dict(tau_min=tau, threshold=threshold, target_reached=reached, calibration=outcome))
    reaching = [s for s in smoothed if s['target_reached']]
    choices['smoothed'] = (min(reaching, key=lambda s: (s['calibration']['mean_completion_min'], s['tau_min'])) if reaching
                           else max(smoothed, key=lambda s: s['calibration']['success_rate']))
    rates = success[:, cal, :].mean(axis=(0, 1))
    hits = np.flatnonzero(rates >= br.Q)
    choices['fixed'] = dict(wait_min=float(br.GRID[hits[0]] if hits.size else br.GRID[-1]), target_reached=bool(hits.size))
    return json.loads(json.dumps(choices))


def evaluate(tables, noise, choices, split=None, observed=None):
    success = br.success_table(tables)
    cal, ev = split_samples(success.shape[1]) if split is None else split
    probability = estimator_probabilities(tables, success, noise, cal, ev, observed)
    out = {name: outcomes(probability[name], success, ev, choices[name]['threshold']) for name in ('current', 'history')}
    out['smoothed'] = outcomes(probability[f"smoothed_{choices['smoothed']['tau_min']:g}"], success, ev,
                               choices['smoothed']['threshold'])
    out['fixed_calibrated'] = fixed_outcome(success, ev, choices['fixed']['wait_min'])
    for wait in br.FIXED_WAITS:
        out[f'fixed_{wait:g}'] = fixed_outcome(success, ev, wait)
    out['oracle'] = outcomes(success[:, ev, :].astype(float), success, ev, 1.)
    return out


def main():
    br.load_protocol()
    load_protocol()
    tables = dict(np.load(br.TABLES))
    success = br.success_table(tables)
    cal, ev = split_samples(success.shape[1])
    noise_out = {}
    for noise in NOISE_LEVELS:
        choices = calibration_choices(tables, noise)
        noise_out[f'{noise:g}'] = dict(choices=choices, evaluation=evaluate(tables, noise, choices))
    curve = [dict(wait_min=float(w), **{k: v for k, v in fixed_outcome(success, ev, w).items() if k != 'mean_wait_min'})
             for w in br.GRID]
    main_block, zero = noise_out[f'{br.NOISE_SD:g}']['evaluation'], noise_out['0']['evaluation']
    statements = dict(
        W1=dict(history=main_block['history'], fixed_calibrated=main_block['fixed_calibrated'],
                completion_difference_min=main_block['fixed_calibrated']['mean_completion_min'] - main_block['history']['mean_completion_min']),
        W2={k: main_block[k] for k in ('history', 'smoothed', 'current')},
        W3=dict(history=zero['history'], current=zero['current'],
                completion_difference_min=zero['current']['mean_completion_min'] - zero['history']['mean_completion_min']))
    result = dict(status='complete', protocol_sha256=protocol_sha256(),
                  frozen_utc=json.loads(FREEZE.read_text(encoding='utf-8'))['frozen_utc'],
                  split=dict(seed=SPLIT_SEED, calibration=cal.tolist(), evaluation=ev.tolist()),
                  noise=noise_out, fixed_wait_curve_eval=curve, statements=statements,
                  limitations=['Simulation of one published model structure; not evidence about measured cells.',
                               'Calibration and evaluation samples share the model structure (no structural mismatch).',
                               'Posterior samples are population-mean uncertainty, not cell heterogeneity.'])
    RESULTS.write_text(json.dumps(result, indent=2) + '\n', newline='\n')
    print(json.dumps(dict(choices={k: v['choices'] for k, v in noise_out.items()},
                          evaluation={k: {p: {m: round(x, 3) for m, x in r.items()} for p, r in v['evaluation'].items()}
                                      for k, v in noise_out.items()}), indent=2))


if __name__ == '__main__':
    main()
