"""Condition map for model-based waiting on B3: feedback delay, sampling interval and noise.

Implements configs/b3_delay_map_protocol.json (revision plan item 8), frozen
before its only run. A probe decided at an observation instant t acts on the
receiver at t + d, so a waiting rule must predict readiness d minutes ahead.
The B3-predictive rule weights reference samples by the observations so far
and averages their success at t + d; the model-free comparator extrapolates
the smoothed reporter level and its increment with a Gaussian kernel over the
same reference samples. Rules are calibrated per history (target 0.90) on
calibration samples and scored on evaluation samples over the 20 splits of
the robustness study. Simulation of one published model structure, not
evidence about measured cells.
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

import b3_readiness as br
import b3_waiting as bw
import b3_waiting_baselines as bl
import b3_waiting_frontier as wf
import b3_waiting_robustness as rb

PROTOCOL = br.ROOT / 'configs/b3_delay_map_protocol.json'
FREEZE = br.OUT / 'b3_delay_map_protocol_freeze.json'
RESULTS = br.OUT / 'delay_map_results.json'
DELAYS = (0, 10, 20, 30, 40)
INTERVALS = (2, 6, 10)
NOISES = (.0025, .005, .01)
SCALES = bw.SMOOTH_TAUS
THRESHOLDS = bl.FINE
FLOOR = bw.LIKELIHOOD_FLOOR
TARGET = .90
BIN = .2          # grid spacing in kernel standard deviations (whitened coordinates)
PAD = 6.
RULES = ('fixed', 'b3_predictive', 'b3_nowcast', 'smoothed_slope')
PAIRS = (('b3_predictive', 'smoothed_slope'), ('b3_predictive', 'fixed'), ('smoothed_slope', 'fixed'),
         ('b3_nowcast', 'b3_predictive'))
HELPS_SPLITS, HELPS_SUCCESS_MARGIN = 16, .005


def protocol_sha256():
    return hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()


def load_protocol():
    if protocol_sha256() != json.loads(FREEZE.read_text(encoding='utf-8'))['protocol_sha256']:
        raise SystemExit('Protocol differs from its freeze record; refusing to run.')
    bw.load_protocol()
    return json.loads(PROTOCOL.read_text(encoding='utf-8'))


def instant_index(interval):
    """Decision-grid indices of the observation instants 0, Delta, 2 Delta, ... <= 120 min."""
    return np.arange(0, int(br.GRID[-1]) + 1, interval) // 2


def valid(instants, d):
    """Instants whose arrival t + d lies on the grid."""
    return instants[instants + d // 2 < br.GRID.size]


def threshold_arrivals(probability, instants, d, thresholds):
    """Arrival index per threshold: first valid instant whose estimate reaches it (+ d), else the last grid time."""
    peak = np.maximum.accumulate(probability, axis=-1)
    hit = peak[None] >= np.asarray(thresholds)[:, None, None, None]
    first = hit.argmax(-1)
    return np.where(hit.any(-1), instants[first] + d // 2, br.GRID.size - 1)


def b3_weights(observed, trajectories, sigma, exclude):
    """Normalized weights (n, R, I): row p weights the references by plant p's observations up to each instant."""
    out = np.empty((observed.shape[0],) + trajectories.shape)
    for start in range(0, observed.shape[0], 100):
        stop = min(start + 100, observed.shape[0])
        log_w = -np.cumsum((trajectories[None] - observed[start:stop, None, :]) ** 2, axis=2) / (2 * sigma ** 2)
        for i, p in enumerate(range(start, stop)):
            if exclude[p] >= 0:
                log_w[i, exclude[p]] = -np.inf
        log_w -= log_w.max(axis=1, keepdims=True)
        weights = np.exp(log_w)
        out[start:stop] = weights / weights.sum(axis=1, keepdims=True)
    return out


def ema_increment(series, a):
    """Exponential smoothing with coefficient a along the last axis, and its one-step increment (0 at the start)."""
    level = np.empty_like(series, dtype=float)
    level[..., 0] = series[..., 0]
    for k in range(1, series.shape[-1]):
        level[..., k] = a * level[..., k - 1] + (1 - a) * series[..., k]
    increment = np.zeros_like(level)
    increment[..., 1:] = np.diff(level, axis=-1)
    return level, increment


def kernel_whitener(noise, a):
    v = noise ** 2 * (1 - a) / (1 + a)
    cov = v * np.array([[1., 1 - a], [1 - a, 2 * (1 - a)]]) + FLOOR ** 2 * np.eye(2)
    return np.linalg.inv(np.linalg.cholesky(cov))


class BinnedKernel:
    """Gaussian-kernel sums over reference states in whitened coordinates (unit kernel s.d.), by linear binning."""

    def __init__(self, points, values, extent):
        self.lo = extent[0] - PAD
        shape = np.ceil((extent[1] + PAD - self.lo) / BIN).astype(int) + 2
        if shape.prod() > 4e7:
            raise RuntimeError(f'kernel grid too large: {shape}')
        self.s0, self.s1 = np.zeros(shape), np.zeros(shape)
        u = (points - self.lo) / BIN
        base = np.floor(u).astype(int)
        frac = u - base
        for dx in (0, 1):
            for dy in (0, 1):
                weight = (frac[:, 0] if dx else 1 - frac[:, 0]) * (frac[:, 1] if dy else 1 - frac[:, 1])
                np.add.at(self.s0, (base[:, 0] + dx, base[:, 1] + dy), weight)
                np.add.at(self.s1, (base[:, 0] + dx, base[:, 1] + dy), weight * values)
        scale = 2 * np.pi / BIN ** 2      # unit-peak kernel exp(-r^2/2) instead of a normalized filter
        self.s0 = gaussian_filter(self.s0, 1 / BIN, mode='constant', truncate=PAD) * scale
        self.s1 = gaussian_filter(self.s1, 1 / BIN, mode='constant', truncate=PAD) * scale

    def sums(self, points):
        coords = ((points - self.lo) / BIN).T
        return (map_coordinates(self.s0, coords, order=1, mode='constant'),
                map_coordinates(self.s1, coords, order=1, mode='constant'))


def smoothed_slope_probabilities(ref_states, ref_success, query_states, own_states, own_success, whitener):
    """Kernel estimate at query states (H, n, I, 2); own_* (n, J, 2)/(n, J) are left out per plant (None: no exclusion)."""
    ref_w, query_w = ref_states @ whitener.T, query_states @ whitener.T
    stacked = np.concatenate([ref_w, query_w.reshape(-1, 2)])
    kernel = BinnedKernel(ref_w, ref_success, (stacked.min(0), stacked.max(0)))
    s0, s1 = kernel.sums(query_w.reshape(-1, 2))
    s0, s1 = s0.reshape(query_w.shape[:-1]), s1.reshape(query_w.shape[:-1])
    if own_states is not None:
        own_w = own_states @ whitener.T
        for p in range(query_w.shape[1]):
            k = np.exp(-((query_w[:, p, :, None, :] - own_w[p][None, None]) ** 2).sum(-1) / 2)
            s0[:, p] -= k.sum(-1)
            s1[:, p] -= k @ own_success[p]
    return np.where(s0 > 1e-9, s1 / np.maximum(s0, 1e-300), 0.)


def fixed_options(success, plants, deadline):
    """One arrival time per history on the grid (no observations)."""
    k = np.broadcast_to(np.arange(br.GRID.size)[:, None, None], (br.GRID.size, success.shape[0], len(plants)))
    ok, completion = wf.scored(success, plants, k, deadline)
    return [(dict(wait_min=float(w)), ok[i], completion[i]) for i, w in enumerate(br.GRID)]


def rule_options(tables, success, observed, noise, interval, references, plants, deadline=br.DEADLINE_MIN):
    """Every parameter of every observation-based rule for every delay: {d: {rule: [(label, ok, completion)]}}."""
    fret = tables['history_fret'].astype(float)
    instants = instant_index(interval)
    position = {int(p): i for i, p in enumerate(references)}
    exclude = np.array([position.get(int(p), -1) for p in plants])
    sigma = max(noise, FLOOR)
    weights = np.stack([b3_weights(observed[h][plants][:, instants], fret[h][references][:, instants], sigma, exclude)
                        for h in range(fret.shape[0])])                       # (H, n, R, I)
    ref_success = success[:, references, :].astype(float)                    # (H, R, K)
    options = {d: {} for d in DELAYS}
    for d in DELAYS:
        usable = valid(instants, d)
        cols = np.arange(usable.size)
        for name, shift in (('b3_predictive', d // 2), ('b3_nowcast', 0)):
            target = ref_success[:, :, usable + shift]                       # (H, R, I_d)
            probability = np.einsum('hnri,hri->hni', weights[..., cols], target)
            ok, completion = wf.scored(success, plants, threshold_arrivals(probability, usable, d, THRESHOLDS), deadline)
            options[d][name] = [(dict(threshold=float(c)), ok[i], completion[i]) for i, c in enumerate(THRESHOLDS)]
        options[d]['smoothed_slope'] = []
    for scale in SCALES:
        a = np.exp(-interval / scale)
        whitener = kernel_whitener(noise, a)
        ref_level, ref_inc = ema_increment(fret[:, references][:, :, instants], a)
        obs_level, obs_inc = ema_increment(observed[:, plants][:, :, instants], a)
        query = np.stack([obs_level, obs_inc], -1)                              # (H, n, I, 2)
        for d in DELAYS:
            usable = valid(instants, d)
            cols = np.arange(usable.size)
            states = np.stack([ref_level[..., cols], ref_inc[..., cols]], -1)    # (H, R, I_d, 2)
            labels = ref_success[:, :, usable + d // 2]                          # (H, R, I_d)
            own_states = own_success = None
            if (exclude >= 0).any():
                own_states = np.stack([states[:, e].reshape(-1, 2) for e in exclude])
                own_success = np.stack([labels[:, e].reshape(-1) for e in exclude])
            probability = smoothed_slope_probabilities(states.reshape(-1, 2), labels.reshape(-1), query[:, :, cols],
                                                       own_states, own_success, whitener)
            ok, completion = wf.scored(success, plants, threshold_arrivals(probability, usable, d, THRESHOLDS), deadline)
            options[d]['smoothed_slope'] += [(dict(tau_min=scale, threshold=float(c)), ok[i], completion[i])
                                             for i, c in enumerate(THRESHOLDS)]
    fixed = fixed_options(success, plants, deadline)
    for d in DELAYS:
        options[d]['fixed'] = fixed
    return options


def evaluate_cell(options_cal, options_ev, oracle):
    choices = {name: {hist: wf.choose(options_cal[name], h, TARGET) for h, hist in enumerate(br.HISTORIES)} for name in RULES}
    outcomes = {}
    for name in RULES:
        parts = [wf.lookup(options_ev[name], choices[name][hist]) for hist in br.HISTORIES]
        outcomes[name] = (np.stack([parts[h][0][h] for h in range(len(parts))]),
                          np.stack([parts[h][1][h] for h in range(len(parts))]))
    rules = {name: rb.summarise(*pair) for name, pair in outcomes.items()}
    rules['oracle'] = rb.summarise(*oracle)
    per_sample = {name: (ok.mean(0), completion.mean(0)) for name, (ok, completion) in outcomes.items()}
    paired = {f'{a}-{b}': dict(success=float((per_sample[a][0] - per_sample[b][0]).mean()),
                               completion=float((per_sample[a][1] - per_sample[b][1]).mean())) for a, b in PAIRS}
    return dict(choices=choices, rules=rules, paired=paired,
                meets_per_history={name: bool(all(rules[name][h]['success_rate'] >= br.Q for h in br.HISTORIES))
                                   for name in RULES})


def run_job(r, noise, interval):
    tables = dict(np.load(br.TABLES))
    success = br.success_table(tables)
    split_seed, noise_seed = rb.seeds(r)
    cal, ev = bw.split_samples(success.shape[1], seed=split_seed)
    observed = bw.noisy_observations(tables, noise, seed=noise_seed)
    options_cal = rule_options(tables, success, observed, noise, interval, cal, cal)
    options_ev = rule_options(tables, success, observed, noise, interval, cal, ev)
    oracle = wf.oracle_outcomes(success, ev, br.DEADLINE_MIN)
    cells = {f'{d}': evaluate_cell(options_cal[d], options_ev[d], oracle) for d in DELAYS}
    return dict(r=r, noise=noise, interval=interval, split_seed=split_seed, noise_seed=noise_seed, cells=cells)


def _job(job):
    return run_job(*job)


def cell_key(noise, interval, d):
    return f'{noise:g}/{interval}/{d}'


def summarize(rows):
    paired = {pair: {key: rb.stats([row['paired'][pair][key] for row in rows]) for key in ('success', 'completion')}
              for pair in rows[0]['paired']}
    faster = int(sum(row['paired']['b3_predictive-smoothed_slope']['completion'] < 0 for row in rows))
    out = dict(paired=paired, b3_faster_than_smoothed_slope=faster,
               faster={pair: int(sum(row['paired'][pair]['completion'] < 0 for row in rows)) for pair in rows[0]['paired']},
               meets_per_history={name: int(sum(row['meets_per_history'][name] for row in rows)) for name in RULES},
               success={name: {part: rb.stats([row['rules'][name][part]['success_rate'] for row in rows])
                               for part in wf.PARTS} for name in RULES + ('oracle',)},
               completion={name: {part: rb.stats([row['rules'][name][part]['mean_completion_min'] for row in rows])
                                  for part in wf.PARTS} for name in RULES + ('oracle',)})
    success_gap = (out['success']['b3_predictive']['all']['median'] - out['success']['smoothed_slope']['all']['median'])
    out['map_value_min'] = paired['b3_predictive-smoothed_slope']['completion']['median']
    out['map_class'] = ('model helps' if faster >= HELPS_SPLITS and success_gap >= -HELPS_SUCCESS_MARGIN
                        else 'reporter suffices')
    return out


def main():
    load_protocol()
    jobs = [(r, noise, interval) for noise in NOISES for interval in INTERVALS for r in range(rb.REPEATS)]
    with ProcessPoolExecutor(max_workers=min(32, os.cpu_count() or 1)) as pool:
        results = list(pool.map(_job, jobs))
    splits = {}
    for (r, noise, interval), result in zip(jobs, results):
        for d, cell in result['cells'].items():
            splits.setdefault(cell_key(noise, interval, int(d)), {})[str(r)] = cell
    summary = {key: summarize([rows[str(r)] for r in range(rb.REPEATS)]) for key, rows in splits.items()}
    result = dict(status='complete', protocol_sha256=protocol_sha256(),
                  frozen_utc=json.loads(FREEZE.read_text(encoding='utf-8'))['frozen_utc'],
                  grid=dict(delays_min=list(DELAYS), intervals_min=list(INTERVALS), noise_sd=list(NOISES)),
                  splits=splits, summary=summary,
                  limitations=['Simulation of one published model structure; the B3 rule shares the plant structure.',
                               'Repeats redraw the split and observation noise only; one finite posterior set is reused.',
                               'Per-history targets are empirical calibration targets, not coverage guarantees.'])
    RESULTS.write_text(json.dumps(result, indent=2) + '\n', newline='\n')
    for key, block in summary.items():
        print(key, block['map_class'], round(block['map_value_min'], 2), block['b3_faster_than_smoothed_slope'],
              {n: round(100 * block['success'][n]['all']['median'], 1) for n in RULES},
              {n: round(block['completion'][n]['all']['median'], 1) for n in RULES})


if __name__ == '__main__':
    main()
