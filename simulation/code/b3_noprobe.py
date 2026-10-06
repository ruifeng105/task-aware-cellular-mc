"""No-probe control for the B3 readiness task (reviewer item R4).

Implements configs/b3_noprobe_protocol.json, frozen before its only run. For
every posterior sample, previous command and wait it simulates the reporter
with and without the probe, and reports how often the original criterion is
met without any probe, how the original success agrees with a causal
(probe-induced) success, and how the version-3 decisions score under the
causal labels. Simulation of one published model structure.
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os

import numpy as np

import b3_model as b3
import b3_readiness as br
import b3_waiting as bw
import b3_waiting_robustness as rb

PROTOCOL = br.ROOT / 'configs/b3_noprobe_protocol.json'
FREEZE = br.OUT / 'b3_noprobe_protocol_freeze.json'
RESULTS = br.OUT / 'noprobe_results.json'
ARRAYS = br.OUT / 'noprobe_tables.npz'
WINDOW = np.arange(0., br.WINDOW_MIN + 1., 1.)
FULL = np.arange(0., br.GRID[-1] + br.WINDOW_MIN + 1., 1.)


def protocol_sha256():
    return hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()


def load_protocol():
    if protocol_sha256() != json.loads(FREEZE.read_text(encoding='utf-8'))['protocol_sha256']:
        raise SystemExit('Protocol differs from its freeze record; refusing to run.')
    return json.loads(PROTOCOL.read_text(encoding='utf-8'))


def window_pair(model, row, history, concentration, wait):
    """Reporter with and without the probe on the 1-min task window starting at `wait`."""
    times = wait + WINDOW
    probe = model.fret_from_states(row, model.simulate_schedule_states(row, list(history) + [(wait, wait + br.PROBE_MIN)],
                                                                       concentration, times))
    noprobe = model.fret_from_states(row, model.simulate_schedule_states(row, list(history), concentration, times))
    return probe, noprobe


def _job(job):
    kind, row, history, concentration, wait = job
    model, row = b3.worker_model(), np.array(row)
    if kind == 'noprobe':
        return model.fret_from_states(row, model.simulate_schedule_states(row, list(history), concentration, FULL))
    times = wait + WINDOW
    return model.fret_from_states(row, model.simulate_schedule_states(row, list(history) + [(wait, wait + br.PROBE_MIN)],
                                                                      concentration, times))


def simulate(protocol, workers=None):
    posterior, dose = b3.load_posterior(), protocol['command_ng_per_ml']
    histories = {name: tuple(tuple(s) for s in protocol['histories'][name]) for name in br.HISTORIES}
    jobs = [('noprobe', tuple(r), histories[n], dose, 0.) for n in br.HISTORIES for r in posterior]
    jobs += [('probe', tuple(r), histories[n], dose, float(w)) for n in br.HISTORIES for r in posterior for w in br.GRID]
    with ProcessPoolExecutor(max_workers=workers or min(32, os.cpu_count() or 1)) as pool:
        out = list(pool.map(_job, jobs, chunksize=256))
    n, waits, h = len(posterior), len(br.GRID), len(br.HISTORIES)
    noprobe = np.array(out[:h * n]).reshape(h, n, len(FULL))
    probe = np.array(out[h * n:]).reshape(h, n, waits, len(WINDOW))
    start = br.GRID.astype(int)
    windows = np.stack([noprobe[:, :, s:s + len(WINDOW)] for s in start], axis=2)
    return dict(noprobe=noprobe, rise_probe=probe.max(-1) - probe[..., 0],
                rise_without_probe=windows.max(-1) - windows[..., 0], probe_increment=(probe - windows).max(-1))


def rescore(tables, causal):
    """Version-3 rules on the primary split at noise 0.005, scored with the original and the causal labels."""
    v3 = json.loads(bw.RESULTS.read_text(encoding='utf-8'))['noise']['0.005']['choices']
    success = br.success_table(tables)
    cal, ev = bw.split_samples(success.shape[1])
    probability = bw.estimator_probabilities(tables, success, .005, cal, ev)
    out = {'original': {}, 'causal': {}}
    for name in ('current', 'smoothed', 'history', 'fixed'):
        if name == 'fixed':
            k = np.full((success.shape[0], len(ev)), int(np.flatnonzero(br.GRID == v3['fixed']['wait_min'])[0]))
        else:
            key = f"smoothed_{v3['smoothed']['tau_min']:g}" if name == 'smoothed' else name
            k = np.apply_along_axis(br.first_crossing, -1, probability[key], v3[name]['threshold'])
        for label, table in (('original', success), ('causal', causal)):
            ok = np.take_along_axis(table[:, ev, :], k[..., None], -1)[..., 0]
            completion = np.where(ok, br.GRID[k] + br.WINDOW_MIN, br.DEADLINE_MIN)
            out[label][name] = dict(success_rate=float(ok.mean()), mean_completion_min=float(completion.mean()))
    return out


def main():
    load_protocol()
    protocol = br.load_protocol()
    tables = dict(np.load(br.TABLES))
    arrays = simulate(protocol)
    np.savez_compressed(ARRAYS, **arrays)
    threshold = br.KAPPA * tables['naive_rise'].astype(float)[None, :, None]
    original = br.success_table(tables)
    without = arrays['rise_without_probe'] >= threshold
    causal = arrays['probe_increment'] >= threshold
    per_history = lambda a: {name: float(a[h].mean()) for h, name in enumerate(br.HISTORIES)}
    result = dict(
        status='complete', protocol_sha256=protocol_sha256(),
        frozen_utc=json.loads(FREEZE.read_text(encoding='utf-8'))['frozen_utc'],
        success_without_probe=dict(fraction=float(without.mean()), per_history=per_history(without),
                                   fraction_of_original_successes=float(without[original].mean()),
                                   max_rise_without_probe_over_naive=float((arrays['rise_without_probe'] / threshold * br.KAPPA).max())),
        agreement=dict(both=int((original & causal).sum()), original_only=int((original & ~causal).sum()),
                       causal_only=int((~original & causal).sum()), neither=int((~original & ~causal).sum()),
                       fraction_agree=float((original == causal).mean())),
        readiness_curves=dict(waits_min=br.GRID.tolist(),
                              original={n: original[h].mean(0).tolist() for h, n in enumerate(br.HISTORIES)},
                              causal={n: causal[h].mean(0).tolist() for h, n in enumerate(br.HISTORIES)},
                              without_probe={n: without[h].mean(0).tolist() for h, n in enumerate(br.HISTORIES)}),
        rescored=rescore(tables, causal),
        limitations=['Simulation of one published model structure; not evidence about measured cells.',
                     'The original task definition remains the primary analysis.'])
    RESULTS.write_text(json.dumps(result, indent=2) + '\n', newline='\n')
    print(json.dumps({k: v for k, v in result.items() if k in ('success_without_probe', 'agreement', 'rescored')}, indent=2))


if __name__ == '__main__':
    main()
