"""Probe responses of the B3 posterior over a 30-min window, for the observation-window sensitivity of E4.

Simulates, with the solver and schedules of b3_readiness, every posterior sample's FRET on a 1-min grid from the
probe onset w to w + 30 min (3- and 30-min histories, w = 0, 2, ..., 120) and the naive response over [0, 30];
stores the rise max F(w..w+W) - F(w) for W = 10, 20 and 30 min. W = 20 must reproduce readiness_tables.npz.
Simulation of one published model structure, not evidence about measured cells.
Usage: python b3_window_tables.py [--time N]   (--time: simulate N jobs and report the time per job)
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')  # one BLAS thread per process: no oversubscription in pools
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import time

import numpy as np

import b3_model as b3
import b3_readiness as br

OUT = br.OUT / 'window_tables.npz'
WINDOWS = (10, 20, 30)
LONGEST = max(WINDOWS)


def _job(job):
    kind, row, history, concentration, wait = job
    model, row = b3.worker_model(), np.array(row)
    times = np.arange(wait, wait + LONGEST + 1., 1.)
    states = model.simulate_schedule_states(row, list(history) + [(wait, wait + br.PROBE_MIN)], concentration, times)
    fret = model.fret_from_states(row, states)
    return [float(fret[:w + 1].max() - fret[0]) for w in WINDOWS]


def jobs():
    protocol = json.loads(br.SIMULATION_PROTOCOL.read_text(encoding='utf-8'))
    posterior, dose = b3.load_posterior(), protocol['command_ng_per_ml']
    histories = {name: tuple(tuple(s) for s in protocol['histories'][name]) for name in br.HISTORIES}
    out = [('naive', tuple(r), (), dose, 0.) for r in posterior]
    out += [('probe', tuple(r), histories[n], dose, float(w)) for n in br.HISTORIES for r in posterior for w in br.GRID]
    return out, len(posterior)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--time', type=int, default=0)
    args = parser.parse_args()
    br.load_protocol()
    work, samples = jobs()
    workers = min(24, os.cpu_count() or 1)
    if args.time:
        rng = np.random.default_rng(0)
        pick = [work[i] for i in rng.choice(len(work), args.time, replace=False)]
        start = time.monotonic()
        with ProcessPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_job, pick, chunksize=1))
        elapsed = time.monotonic() - start
        print(f'{args.time} jobs in {elapsed:.1f} s on {workers} workers; estimated total '
              f'{elapsed / args.time * len(work) / 60:.1f} min for {len(work)} jobs')
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        out = np.array(list(pool.map(_job, work, chunksize=64)), dtype=np.float32)
    naive = out[:samples].T                                                  # (3, samples)
    rise = out[samples:].reshape(len(br.HISTORIES), samples, br.GRID.size, len(WINDOWS)).transpose(3, 0, 1, 2)
    np.savez_compressed(OUT, windows=np.array(WINDOWS), naive_rise=naive, rise=rise)
    tables = dict(np.load(br.TABLES))
    w20 = WINDOWS.index(20)
    print('W=20 max |difference| to readiness tables: naive',
          float(np.abs(naive[w20] - tables['naive_rise']).max()), 'probe', float(np.abs(rise[w20] - tables['rise']).max()))


if __name__ == '__main__':
    main()
