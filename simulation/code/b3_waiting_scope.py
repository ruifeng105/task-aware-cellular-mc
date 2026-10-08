"""E4: sensitivity of the strict waiting comparison to the task definition and to the form of model mismatch.

Implements configs/b3_waiting_scope_protocol.json, frozen before its only run. Part A re-runs the certified arm
of E1 one factor at a time for the recovery fraction kappa, the observation window W (probe-window tables of
b3_window_tables) and the failure penalty D. Part B scores the E1 rules, and rules re-certified on true outcomes,
when the plant's recovery after the 30-min command is shifted, scaled in amplitude or slowed, when receivers
differ in recovery speed, and at amplitude and speed anchored to the single measured mixed-protocol probe. All
mismatch forms are stress tests hidden from the reporter; only the anchored variants use a measurement.
Simulation of one published model structure, not evidence about measured cells.
Usage: python b3_waiting_scope.py
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')  # one BLAS thread per process: no oversubscription in pools
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json

import numpy as np

import b3_readiness as br
import b3_waiting as bw
import b3_waiting_frontier as fr
import b3_waiting_strict as st
import natural_probe as npb
import strict_eval as se

PROTOCOL = br.ROOT / 'configs/b3_waiting_scope_protocol.json'
FREEZE = br.OUT / 'b3_waiting_scope_protocol_freeze.json'
RESULTS = br.OUT / 'waiting_scope_results.json'
WINDOW_TABLES = br.OUT / 'window_tables.npz'
NOISE, ALPHA = .005, .025
TASK_CELLS = {'baseline': (.5, 20, 140.), 'kappa_0.4': (.4, 20, 140.), 'kappa_0.6': (.6, 20, 140.),
              'kappa_0.8': (.8, 20, 140.), 'window_10': (.5, 10, 130.), 'window_30': (.5, 30, 150.),
              'penalty_200': (.5, 20, 200.), 'penalty_260': (.5, 20, 260.)}
MISMATCH = {'shift': (4., 8., 16.), 'amplitude': (.9, .8, .7), 'rate': (1.1, 1.25, 1.5), 'heterogeneity': (.1, .2)}
ANCHORED = ('anchored_amplitude', 'anchored_rate')
HET_SEED = 20261016
ARMS = ('model_calibrated', 'recalibrated')
LONG = br.HISTORIES.index('long')


def protocol_sha256():
    return hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()


def load_protocol():
    if protocol_sha256() != json.loads(FREEZE.read_text(encoding='utf-8'))['protocol_sha256']:
        raise SystemExit('Protocol differs from its freeze record; refusing to run.')
    st.load_protocol()
    return json.loads(PROTOCOL.read_text(encoding='utf-8'))


def task_tables(tables, window):
    """Rise and naive rise for window W; W = 20 uses the readiness tables of every earlier study."""
    if window == 20:
        return tables['rise'], tables['naive_rise']
    data = dict(np.load(WINDOW_TABLES))
    i = int(np.flatnonzero(data['windows'] == window)[0])
    return data['rise'][i], data['naive_rise'][i]


def scored(success, plants, k, window, deadline):
    sub = np.broadcast_to(success[:, plants, :], k.shape + (success.shape[-1],))
    ok = np.take_along_axis(sub, k[..., None], -1)[..., 0]
    return ok, np.where(ok, br.GRID[k] + window, deadline)


def options_and_waits(probability, success, plants, window, deadline):
    """As b3_waiting_strict.options_and_waits with completion = wait + window on success."""
    options, waits = {}, {}

    def add(name, label, k):
        ok, completion = scored(success, plants, k, window, deadline)
        options.setdefault(name, []).append((label, ok, completion))
        waits[(name, se.tie_key(label))] = br.GRID[k]

    for name in ('current', 'history'):
        k = fr.crossing_index(probability[name], fr.THRESHOLDS)
        for i, c in enumerate(fr.THRESHOLDS):
            add(name, dict(threshold=float(c)), k[i])
    for tau in bw.SMOOTH_TAUS:
        k = fr.crossing_index(probability[f'smoothed_{tau:g}'], fr.THRESHOLDS)
        for i, c in enumerate(fr.THRESHOLDS):
            add('smoothed', dict(tau_min=tau, threshold=float(c)), k[i])
    for j, w in enumerate(br.GRID):
        add('fixed', dict(wait_min=float(w)), np.full((success.shape[0], len(plants)), j))
    return options, waits


def certify(options):
    return {name: {hist: se.certified_choice(st.chains(options, name), h, br.Q, ALPHA)
                   for h, hist in enumerate(br.HISTORIES)} for name in st.RULES}


def score(options, waits, choices, success, plants, window, deadline):
    outcomes = {}
    for name in st.RULES:
        parts = [st.action(options, waits, name, choices[name][hist]) for hist in br.HISTORIES]
        outcomes[name] = tuple(np.stack([parts[h][i][h] for h in range(len(parts))]) for i in range(3))
    k = fr.crossing_index(success[:, plants, :].astype(float), [1.])[0]
    oracle_ok, oracle_completion = scored(success, plants, k, window, deadline)
    return dict(rules={name: st.per_history(*outcomes[name]) for name in st.RULES}, paired=st.paired(outcomes),
                oracle=st.per_history(oracle_ok, oracle_completion, br.GRID[k]))


def run_cell(r, cell, tables=None):
    tables = dict(np.load(br.TABLES)) if tables is None else tables
    kappa, window, deadline = TASK_CELLS[cell]
    rise, naive = task_tables(tables, window)
    success = rise >= kappa * naive[None, :, None]
    R, C, T = st.roles(r)
    observed = bw.noisy_observations(tables, NOISE, seed=st.noise_seed(r))
    options_c, _ = options_and_waits(bw.estimator_probabilities(tables, success, NOISE, R, C, observed),
                                     success, C, window, deadline)
    options_t, waits_t = options_and_waits(bw.estimator_probabilities(tables, success, NOISE, R, T, observed),
                                           success, T, window, deadline)
    choices = certify(options_c)
    return dict(r=r, cell=cell, choices=choices, **score(options_t, waits_t, choices, success, T, window, deadline))


def rho_table(tables):
    return tables['rise'].astype(float) / tables['naive_rise'].astype(float)[None, :, None]


def stretched(rho_rows, factors):
    """rho at decision time t / s per row (linear interpolation on the decision grid); factors per row."""
    out = np.empty_like(rho_rows)
    for i, (row, s) in enumerate(zip(rho_rows, factors)):
        out[i] = np.interp(br.GRID / s, br.GRID, row)
    return out


def anchors(tables):
    """Amplitude and speed factors that bring the posterior-median recovery after the 30-min command to the
    measured mixed-probe ratio at 60 min (single measured point)."""
    measured = json.loads(npb.RESULTS.read_text(encoding='utf-8'))['probes']['fgf_mixed_25ng/pulse_3']['rise_over_naive']['measured_median']
    curve = np.median(rho_table(tables)[LONG], axis=0)
    k60 = int(np.flatnonzero(br.GRID == 60.)[0])
    j = max(i for i in range(k60) if curve[i] <= measured)
    crossing = br.GRID[j] + (measured - curve[j]) / (curve[j + 1] - curve[j]) * (br.GRID[j + 1] - br.GRID[j])
    return dict(measured_ratio=float(measured), median_ratio_at_60=float(curve[k60]),
                amplitude=float(measured / curve[k60]), crossing_min=float(crossing), rate=float(60. / crossing))


def plant_success(tables, scenario, level):
    """Plant outcomes (H, samples, waits) under one mismatch form; the reporter is never changed."""
    model = br.success_table(tables)
    rho = rho_table(tables)
    plant = model.copy()
    if scenario == 'shift':
        steps = int(level / 2.)
        if steps:
            plant[LONG, :, steps:] = model[LONG, :, :-steps]
            plant[LONG, :, :steps] = False
    elif scenario in ('amplitude', 'anchored_amplitude'):
        plant[LONG] = level * rho[LONG] >= br.KAPPA
    elif scenario in ('rate', 'anchored_rate'):
        plant[LONG] = stretched(rho[LONG], np.full(rho.shape[1], level)) >= br.KAPPA
    elif scenario == 'heterogeneity':
        factors = np.exp(level * np.random.default_rng(HET_SEED).standard_normal(rho.shape[1]))
        plant = np.stack([stretched(rho[h], factors) >= br.KAPPA for h in range(rho.shape[0])])
    else:
        raise ValueError(scenario)
    return plant


def scenarios(tables):
    levels = [(name, float(v)) for name, values in MISMATCH.items() for v in values]
    a = anchors(tables)
    return levels + [('anchored_amplitude', a['amplitude']), ('anchored_rate', a['rate'])]


def run_mismatch(r, tables=None, plants=None):
    tables = dict(np.load(br.TABLES)) if tables is None else tables
    model = br.success_table(tables)
    R, C, T = st.roles(r)
    observed = bw.noisy_observations(tables, NOISE, seed=st.noise_seed(r))
    on_c = bw.estimator_probabilities(tables, model, NOISE, R, C, observed)
    on_t = bw.estimator_probabilities(tables, model, NOISE, R, T, observed)
    window, deadline = 20, br.DEADLINE_MIN
    model_choices = certify(options_and_waits(on_c, model, C, window, deadline)[0])
    out = dict(r=r, model_choices=model_choices, scenarios={})
    for scenario, level in scenarios(tables):
        key = f'{scenario}/{level:g}'
        plant = plant_success(tables, scenario, level) if plants is None else plants[key]
        options_t, waits_t = options_and_waits(on_t, plant, T, window, deadline)
        block = dict(model_calibrated=score(options_t, waits_t, model_choices, plant, T, window, deadline))
        if scenario not in ANCHORED:
            choices = certify(options_and_waits(on_c, plant, C, window, deadline)[0])
            block['recalibrated'] = dict(choices=choices,
                                         **score(options_t, waits_t, choices, plant, T, window, deadline))
        out['scenarios'][key] = block
    return out


def _job(job):
    kind, r, cell = job
    return run_cell(r, cell) if kind == 'cell' else run_mismatch(r)


def meets(rules, name):
    return all(rules[name][h]['success_rate'] >= br.Q for h in br.HISTORIES)


def summarize(cells, mismatch):
    task = {}
    for cell in TASK_CELLS:
        rows = [cells[cell][str(r)] for r in range(st.SPLITS)]
        task[cell] = dict(
            meets_per_history={n: int(sum(meets(row['rules'], n) for row in rows)) for n in st.RULES},
            certified_both={n: int(sum(all(row['choices'][n][h]['certified'] for h in br.HISTORIES) for row in rows))
                            for n in st.RULES},
            oracle_feasible=int(sum(all(row['oracle'][h]['success_rate'] >= br.Q for h in br.HISTORIES) for row in rows)),
            completion={n: st.stats([row['rules'][n]['all']['mean_completion_min'] for row in rows]) for n in st.RULES},
            paired_completion={n: st.stats([row['paired'][n]['all']['completion'] for row in rows]) for n in st.ADAPTIVE},
            faster={n: int(sum(row['paired'][n]['all']['completion'] < 0 for row in rows)) for n in st.ADAPTIVE},
            success_long={n: st.stats([row['rules'][n]['long']['success_rate'] for row in rows]) for n in st.RULES})
    forms = {}
    for key in mismatch['0']['scenarios']:
        forms[key] = {}
        oracle = [mismatch[str(r)]['scenarios'][key]['model_calibrated']['oracle'] for r in range(st.SPLITS)]
        forms[key]['oracle'] = dict(feasible=int(sum(all(o[h]['success_rate'] >= br.Q for h in br.HISTORIES) for o in oracle)),
                                    success_long=st.stats([o['long']['success_rate'] for o in oracle]))
        for arm in ARMS:
            if arm not in mismatch['0']['scenarios'][key]:
                continue
            rows = [mismatch[str(r)]['scenarios'][key][arm] for r in range(st.SPLITS)]
            forms[key][arm] = dict(
                meets_per_history={n: int(sum(meets(row['rules'], n) for row in rows)) for n in st.RULES},
                success_long={n: st.stats([row['rules'][n]['long']['success_rate'] for row in rows]) for n in st.RULES},
                completion={n: st.stats([row['rules'][n]['all']['mean_completion_min'] for row in rows]) for n in st.RULES},
                paired_completion={n: st.stats([row['paired'][n]['all']['completion'] for row in rows]) for n in st.ADAPTIVE},
                faster={n: int(sum(row['paired'][n]['all']['completion'] < 0 for row in rows)) for n in st.ADAPTIVE})
            if arm == 'recalibrated':
                forms[key][arm]['certified_both'] = {
                    n: int(sum(all(row['choices'][n][h]['certified'] for h in br.HISTORIES) for row in rows)) for n in st.RULES}
    return dict(task=task, mismatch=forms)


def statements(summary, anchor):
    task, forms = summary['task'], summary['mismatch']
    return {
        'E4-S1': {cell: dict(meets_per_history=b['meets_per_history'], oracle_feasible=b['oracle_feasible'],
                             paired_completion_median={n: b['paired_completion'][n]['median'] for n in st.ADAPTIVE},
                             faster=b['faster']) for cell, b in task.items()},
        'E4-S2': {key: dict(model_calibrated=dict(meets_per_history=b['model_calibrated']['meets_per_history'],
                                                  success_long_median={n: v['median'] for n, v in b['model_calibrated']['success_long'].items()}),
                            recalibrated=dict(meets_per_history=b['recalibrated']['meets_per_history'],
                                              paired_completion_median={n: v['median'] for n, v in b['recalibrated']['paired_completion'].items()},
                                              certified_both=b['recalibrated']['certified_both']),
                            oracle_feasible=b['oracle']['feasible'])
                  for key, b in forms.items() if 'recalibrated' in b},
        'E4-S3': dict(anchors=anchor,
                      **{key: dict(oracle_success_long_median=b['oracle']['success_long']['median'],
                                   oracle_feasible=b['oracle']['feasible'],
                                   model_calibrated_success_long_median={n: v['median'] for n, v in b['model_calibrated']['success_long'].items()})
                         for key, b in forms.items() if key.split('/')[0] in ANCHORED})}


def main():
    load_protocol()
    tables = dict(np.load(br.TABLES))
    jobs = [('cell', r, cell) for cell in TASK_CELLS for r in range(st.SPLITS)] + [('mismatch', r, None) for r in range(st.SPLITS)]
    with ProcessPoolExecutor(max_workers=min(20, os.cpu_count() or 1)) as pool:
        results = list(pool.map(_job, jobs))
    cells, mismatch = {}, {}
    for (kind, r, cell), result in zip(jobs, results):
        if kind == 'cell':
            cells.setdefault(cell, {})[str(r)] = result
        else:
            mismatch[str(r)] = result
    summary = summarize(cells, mismatch)
    anchor = anchors(tables)
    result = dict(status='complete', protocol_sha256=protocol_sha256(),
                  frozen_utc=json.loads(FREEZE.read_text(encoding='utf-8'))['frozen_utc'],
                  anchors=anchor, cells=cells, mismatch=mismatch, summary=summary,
                  statements=statements(summary, anchor),
                  limitations=['Simulation of one published model structure; not evidence about measured cells.',
                               'Mismatch forms are stress tests hidden from the reporter; only the anchored variants use a measurement, a single probe.',
                               'Receiver heterogeneity is an arbitrary log-normal speed factor, not a fitted cell-to-cell model.'])
    RESULTS.write_text(json.dumps(result, indent=2) + '\n', newline='\n')
    for cell, b in summary['task'].items():
        print(cell, 'meets', b['meets_per_history'], 'oracle', b['oracle_feasible'], 'faster', b['faster'],
              'dT', {k: round(v['median'], 2) for k, v in b['paired_completion'].items()})
    for key, b in summary['mismatch'].items():
        print(key, 'oracle', b['oracle'], {arm: (b[arm]['meets_per_history'],
                                                {k: round(v['median'], 2) for k, v in b[arm]['paired_completion'].items()})
                                           for arm in ARMS if arm in b})
    print(anchor)


if __name__ == '__main__':
    main()
