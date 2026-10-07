"""Natural probes in measured cells: real-data check of the B3-based readiness estimators.

Implements configs/natural_probe_protocol.json (revision plan item 10),
frozen before its only run. Later pulses of the 3/20 and mixed protocols at
25 ng/ml act as probes at fixed waits after a known previous command. For
each measured cell the readiness at the probe is predicted from B3 posterior
samples simulated under the recorded schedule (constant, current, smoothed
and history-weighted estimators) and compared with whether the cell's own
probe rise reached kappa times the naive rise. Fixed probe times allow no
completion-time comparison; the check scores predictions and ready calls.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

import b3_forecast as bf
import expanded_fgf2 as e
import nested_extensions as nx

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'configs/natural_probe_protocol.json'
OUT = ROOT / 'results/b3'
FREEZE = OUT / 'natural_probe_protocol_freeze.json'
RESULTS = OUT / 'natural_probe_results.json'
KAPPAS = (.4, .5, .6)
WINDOW_MIN, SMOOTH_MIN, SD_FLOOR = 20., 16., .005
THRESHOLDS = (.5, .9)
PULSES_3_20 = (0., 23., 46., 69., 92., 115.)
MIXED_ONSET = 114.
KEYS = ('fgf_3_20_25ng', 'fgf_mixed_25ng', 'fgf_sp_5_25ng')


def protocol_sha256():
    return hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()


def load_protocol():
    if protocol_sha256() != json.loads(FREEZE.read_text(encoding='utf-8'))['protocol_sha256']:
        raise SystemExit('Protocol differs from its freeze record; refusing to run.')
    return json.loads(PROTOCOL.read_text(encoding='utf-8'))


def moving_average(y):
    """Centred 3-frame moving average along the last axis (2 frames at the ends)."""
    padded = np.concatenate([y[..., :1] * np.nan, y, y[..., -1:] * np.nan], axis=-1)
    stack = np.stack([padded[..., :-2], padded[..., 1:-1], padded[..., 2:]])
    return np.nanmean(stack, axis=0)


def rise(times, y, onset, pre_default):
    """Rise of the 3-frame moving average within (onset, onset + 20] above the pre-probe level; y (..., frames)."""
    before = np.flatnonzero(times <= onset)
    pre = y[..., before[-2:]].mean(-1) if before.size else np.broadcast_to(pre_default, y.shape[:-1])
    window = (times > onset) & (times <= onset + WINDOW_MIN)
    return moving_average(y)[..., window].max(-1) - pre


def last_index(times, onset):
    return int(np.flatnonzero(times <= onset)[-1])


def ema(y, a):
    out = np.empty_like(y, dtype=float)
    out[..., 0] = y[..., 0]
    for k in range(1, y.shape[-1]):
        out[..., k] = a * out[..., k - 1] + (1 - a) * y[..., k]
    return out


def kernel_mean(values, references, success, sd):
    w = np.exp(-(values[:, None] - references[None, :]) ** 2 / (2 * sd ** 2))
    s0 = w.sum(1)
    return np.where(s0 > 1e-300, (w @ success) / np.maximum(s0, 1e-300), success.mean())


def estimates(times, measured, simulated, b3_success, i_star, s_l):
    """Readiness per measured cell from B3 samples: constant, current, smoothed, history."""
    a = np.exp(-(times[1] - times[0]) / SMOOTH_MIN)
    smooth_sd = max(s_l * np.sqrt((1 - a) / (1 + a)), SD_FLOOR)
    success = b3_success.astype(float)
    log_w = -((measured[:, None, :i_star + 1] - simulated[None, :, :i_star + 1]) ** 2).sum(-1) / (2 * s_l ** 2)
    log_w -= log_w.max(1, keepdims=True)
    w = np.exp(log_w)
    history = (w / w.sum(1, keepdims=True)) @ success
    ess = 1. / ((w / w.sum(1, keepdims=True)) ** 2).sum(1)
    return dict(constant=np.full(measured.shape[0], success.mean()),
                current=kernel_mean(measured[:, i_star], simulated[:, i_star], success, s_l),
                smoothed=kernel_mean(ema(measured, a)[:, i_star], ema(simulated, a)[:, i_star], success, smooth_sd),
                history=history), ess


def auc(score, outcome):
    pos, neg = score[outcome], score[~outcome]
    if not pos.size or not neg.size:
        return None
    greater = (pos[:, None] > neg[None, :]).mean()
    ties = (pos[:, None] == neg[None, :]).mean()
    return float(greater + .5 * ties)


def scores(predictions, outcome):
    out = dict(n_cells=int(outcome.size), observed_success=float(outcome.mean()), estimators={})
    for name, p in predictions.items():
        row = dict(mean_prediction=float(p.mean()), brier=float(((p - outcome) ** 2).mean()), auc=auc(p, outcome),
                   ready_calls={})
        for c in THRESHOLDS:
            ready = p >= c
            row['ready_calls'][f'{c:g}'] = dict(
                n_ready=int(ready.sum()), success_if_ready=float(outcome[ready].mean()) if ready.any() else None,
                success_if_waiting=float(outcome[~ready].mean()) if (~ready).any() else None)
        out['estimators'][name] = row
    out['brier_observed_fraction'] = float(outcome.mean() * (1 - outcome.mean()))
    return out


def compute():
    protocol = e.load_protocol()
    blocks, _ = e.load_blocks(protocol)
    blocks = {k: blocks[k] for k in KEYS}
    floor = nx.noise_floor(blocks)
    population = bf.simulate_population(blocks)
    frames = {k: np.asarray(blocks[k]['time'], dtype=float) for k in KEYS}
    measured = {k: np.asarray(blocks[k]['y'], dtype=float) for k in KEYS}
    simulated = {k: population[k][:, frames[k].astype(int)] for k in KEYS}
    start = {k: population[k][:, 0] for k in KEYS}
    complete = {k: np.isfinite(measured[k]).all(1) for k in KEYS}

    naive = {'fgf_3_20_25ng': (rise(frames['fgf_3_20_25ng'], measured['fgf_3_20_25ng'], 0., 1.),
                               rise(frames['fgf_3_20_25ng'], simulated['fgf_3_20_25ng'], 0., start['fgf_3_20_25ng'])),
             'fgf_mixed_25ng': (np.nanmedian(rise(frames['fgf_sp_5_25ng'], measured['fgf_sp_5_25ng'], 0., 1.)[complete['fgf_sp_5_25ng']]),
                                float(np.median(rise(frames['fgf_sp_5_25ng'], simulated['fgf_sp_5_25ng'], 0., start['fgf_sp_5_25ng']))))}
    probes = [('fgf_3_20_25ng', f'pulse_{k + 1}', onset) for k, onset in enumerate(PULSES_3_20) if k]
    probes.append(('fgf_mixed_25ng', 'pulse_3', MIXED_ONSET))
    result = dict(status='complete', protocol_sha256=protocol_sha256(),
                  frozen_utc=json.loads(FREEZE.read_text(encoding='utf-8'))['frozen_utc'],
                  noise_floor=floor, cells_complete={k: int(v.sum()) for k, v in complete.items()},
                  naive_reference=dict(mixed_measured_median=float(naive['fgf_mixed_25ng'][0]),
                                       mixed_b3_median=naive['fgf_mixed_25ng'][1],
                                       three_twenty_measured_median=float(np.nanmedian(naive['fgf_3_20_25ng'][0])),
                                       three_twenty_b3_median=float(np.median(naive['fgf_3_20_25ng'][1]))),
                  probes={}, pooled_3_20={})
    pooled = {f'{k:g}': dict(predictions={}, outcome=[]) for k in KAPPAS}
    for key, name, onset in probes:
        t, y, f = frames[key], measured[key][complete[key]], simulated[key]
        naive_y, naive_f = naive[key]
        if key == 'fgf_3_20_25ng':
            naive_y = naive_y[complete[key]]
        rise_y, rise_f = rise(t, y, onset, 1.), rise(t, f, onset, start[key])
        i_star = last_index(t, onset)
        s_l = max(floor[key], SD_FLOOR)
        block = dict(onset_min=onset, wait_min=20. if key == 'fgf_3_20_25ng' else onset - 54.,
                     decision_frame_min=float(t[i_star]), likelihood_sd=s_l,
                     rise_over_naive=dict(measured_median=float(np.median(rise_y / naive_y)),
                                          b3_median=float(np.median(rise_f / naive_f))),
                     kappa={})
        for kappa in KAPPAS:
            outcome = rise_y >= kappa * naive_y
            b3_success = rise_f >= kappa * naive_f
            predictions, ess = estimates(t, y, f, b3_success, i_star, s_l)
            block['kappa'][f'{kappa:g}'] = dict(b3_population_readiness=float(b3_success.mean()),
                                                median_history_ess=float(np.median(ess)), **scores(predictions, outcome))
            if key == 'fgf_3_20_25ng':
                pooled[f'{kappa:g}']['outcome'].append(outcome)
                for n, p in predictions.items():
                    pooled[f'{kappa:g}']['predictions'].setdefault(n, []).append(p)
        result['probes'][f'{key}/{name}'] = block
    for kappa, block in pooled.items():
        result['pooled_3_20'][kappa] = scores({n: np.concatenate(p) for n, p in block['predictions'].items()},
                                              np.concatenate(block['outcome']))
    result['limitations'] = ['One dose in two recorded protocols with fixed probe times; no completion-time comparison.',
                             'The mixed-protocol naive reference comes from different cells (single 5-min pulse).',
                             'Cells of one condition share a session; 3/20 cells repeat across pulses.']
    return result


def main():
    load_protocol()
    result = compute()
    RESULTS.write_text(json.dumps(result, indent=2) + '\n', newline='\n')
    print(json.dumps(result['naive_reference'], indent=1), result['cells_complete'])
    for name, block in result['probes'].items():
        k = block['kappa']['0.5']
        print(name, 'obs', round(k['observed_success'], 3), 'b3', round(k['b3_population_readiness'], 3),
              'rise/naive', {a: round(b, 3) for a, b in block['rise_over_naive'].items()},
              {n: (round(r['mean_prediction'], 3), round(r['brier'], 3), r['auc'] and round(r['auc'], 3))
               for n, r in k['estimators'].items()})


if __name__ == '__main__':
    main()
