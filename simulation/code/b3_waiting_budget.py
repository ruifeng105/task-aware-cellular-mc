"""E2: calibration of the per-history waiting rules from one probe outcome per calibration receiver.

Implements configs/b3_waiting_budget_protocol.json, frozen before its only run. Uses the receivers, roles and
noise of E1 (b3_waiting_strict). Each rule's parameter is restricted to a 20-rung ladder; every calibration
receiver is probed once, at its decision time under a randomly assigned rung, and only that outcome is returned.
Rungs are chosen by a monotone plug-in fit or by fixed-sequence binomial tests and scored on the test receivers,
next to the same choice made from complete outcome curves (ideal information) at the same budget.
Simulation of one published model structure, not evidence about measured cells.
Usage: python b3_waiting_budget.py
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')  # one BLAS thread per process: no oversubscription in pools
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import hashlib
import json

import numpy as np

import b3_readiness as br
import b3_waiting as bw
import b3_waiting_frontier as fr
import b3_waiting_strict as st
import strict_eval as se

PROTOCOL = br.ROOT / 'configs/b3_waiting_budget_protocol.json'
FREEZE = br.OUT / 'b3_waiting_budget_protocol_freeze.json'
RESULTS = br.OUT / 'waiting_budget_results.json'
NOISE, BUDGETS, RUNGS, ALPHA = .005, (50, 100, 200), 20, .025
RULES = ('fixed', 'current', 'smoothed', 'history', 'history_delay')
ADAPTIVE = RULES[1:]
INFOS, SELECTIONS, SCENARIOS = ('one_probe', 'full_curve'), ('plugin', 'certified'), ('baseline', 'hidden_lag_8')
THRESHOLDS = np.round(.99 - .01 * np.arange(RUNGS), 2)        # 0.99, 0.98, ..., 0.80
WAITS = 120. - 4. * np.arange(RUNGS)                            # 120, 116, ..., 44 min
DELAYS = 38. - 2. * np.arange(RUNGS)                            # 38, 36, ..., 0 min
DELAY_TRIGGER, SMOOTH_TAU, LAG_MIN = .90, 16., 8.
DRAW_SEED, RUNG_SEED = 20261013, 20261014


def protocol_sha256():
    return hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()


def load_protocol():
    if protocol_sha256() != json.loads(FREEZE.read_text(encoding='utf-8'))['protocol_sha256']:
        raise SystemExit('Protocol differs from its freeze record; refusing to run.')
    st.load_protocol()
    return json.loads(PROTOCOL.read_text(encoding='utf-8'))


class LabelOracle:
    """One probe outcome per receiver and history: a second request for the same receiver raises."""

    def __init__(self, success, plants):
        self.success, self.plants, self.used, self.calls = success, np.asarray(plants), set(), 0

    def probe(self, h, i, k):
        if (h, int(i)) in self.used:
            raise RuntimeError(f'receiver {i} was already probed for history {h}')
        self.used.add((h, int(i)))
        self.calls += 1
        return bool(self.success[h, self.plants[i], k])


def plant_success(model, scenario):
    if scenario == 'baseline':
        return model
    shift = int(LAG_MIN / 2.)
    plant, h = model.copy(), br.HISTORIES.index('long')
    plant[h, :, shift:] = model[h, :, :-shift]
    plant[h, :, :shift] = False
    return plant


def ladder_indices(probability, rule, histories, n):
    """Decision indices (RUNGS, H, n), rung 0 the most conservative."""
    if rule == 'fixed':
        return np.broadcast_to((WAITS / 2.).astype(int)[:, None, None], (RUNGS, histories, n)).copy()
    if rule in ('current', 'history'):
        return fr.crossing_index(probability[rule], THRESHOLDS)
    if rule == 'smoothed':
        return fr.crossing_index(probability[f'smoothed_{SMOOTH_TAU:g}'], THRESHOLDS)
    trigger = fr.crossing_index(probability['history'], [DELAY_TRIGGER])[0]
    return np.minimum(trigger[None] + (DELAYS / 2.).astype(int)[:, None, None], br.GRID.size - 1)


def pava_nonincreasing(values, weights):
    """Weighted least-squares non-increasing fit (pool adjacent violators)."""
    blocks = []
    for v, w in zip(values, weights):
        blocks.append([float(v), float(w), 1])
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:
            v2, w2, c2 = blocks.pop()
            v1, w1, c1 = blocks.pop()
            blocks.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, c1 + c2])
    return np.array([b[0] for b in blocks for _ in range(b[2])])


def plugin_one_probe(positions, labels):
    rungs = np.unique(positions)
    means = np.array([labels[positions == j].mean() for j in rungs])
    counts = np.array([(positions == j).sum() for j in rungs], dtype=float)
    meeting = rungs[pava_nonincreasing(means, counts) >= br.Q]
    return int(meeting.max()) if meeting.size else None


def certified_one_probe(positions, labels):
    last = None
    for j in range(RUNGS):
        support = positions >= j
        if not support.any() or se.binom_pvalue(int(labels[support].sum()), int(support.sum()), br.Q) > ALPHA:
            break
        last = j
    return last


def plugin_full(ok):
    meeting = np.flatnonzero(ok.mean(1) >= br.Q)
    return int(meeting.max()) if meeting.size else None


def certified_full(ok):
    m = se.fixed_sequence([(int(row.sum()), int(row.size)) for row in ok], br.Q, ALPHA)
    return m - 1 if m else None


SELECT = {('one_probe', 'plugin'): plugin_one_probe, ('one_probe', 'certified'): certified_one_probe,
          ('full_curve', 'plugin'): plugin_full, ('full_curve', 'certified'): certified_full}


def run_split(r, scenario, tables=None, plant=None):
    tables = dict(np.load(br.TABLES)) if tables is None else tables
    model = br.success_table(tables)
    plant = plant_success(model, scenario) if plant is None else plant
    R, C, T = st.roles(r)
    observed = bw.noisy_observations(tables, NOISE, seed=st.noise_seed(r))
    on_c = bw.estimator_probabilities(tables, model, NOISE, R, C, observed)
    on_t = bw.estimator_probabilities(tables, model, NOISE, R, T, observed)
    histories = len(br.HISTORIES)
    k_c = {rule: ladder_indices(on_c, rule, histories, len(C)) for rule in RULES}
    k_t = {rule: ladder_indices(on_t, rule, histories, len(T)) for rule in RULES}
    rungs = [np.random.default_rng(RUNG_SEED + 1000 * r + h).integers(0, RUNGS, len(C)) for h in range(histories)]
    orders = [np.random.default_rng(DRAW_SEED + 1000 * r + h).permutation(len(C)) for h in range(histories)]
    budgets = {}
    for n in BUDGETS:
        chosen = {(info, sel): {rule: {} for rule in RULES} for info, sel in SELECT}
        calls = {rule: {} for rule in RULES}
        for rule, (h, hist) in product(RULES, enumerate(br.HISTORIES)):
            idx = orders[h][:n]
            positions = rungs[h][idx]
            oracle = LabelOracle(plant, C)
            labels = np.array([oracle.probe(h, i, int(k_c[rule][j, h, i])) for i, j in zip(idx, positions)])
            calls[rule][hist] = oracle.calls
            full = plant[h][C[idx][None, :], k_c[rule][:, h, idx]]
            for info, sel in SELECT:
                data = (positions, labels) if info == 'one_probe' else (full,)
                chosen[(info, sel)][rule][hist] = SELECT[(info, sel)](*data)
        block = dict(probe_calls=calls)
        for info in INFOS:
            block[info] = {}
            for sel in SELECTIONS:
                picks = chosen[(info, sel)]
                outcomes = {}
                for rule in RULES:
                    k = np.stack([k_t[rule][picks[rule][hist], h] if picks[rule][hist] is not None
                                  else np.full(len(T), br.GRID.size - 1) for h, hist in enumerate(br.HISTORIES)])
                    ok, completion = fr.scored(plant, T, k, br.DEADLINE_MIN)
                    outcomes[rule] = (ok, completion, br.GRID[k])
                fixed = outcomes['fixed'][1].mean(0)
                block[info][sel] = dict(
                    choices={rule: {hist: dict(rung=picks[rule][hist], fallback=picks[rule][hist] is None,
                                               certified=(picks[rule][hist] is not None) if sel == 'certified' else None)
                                    for hist in br.HISTORIES} for rule in RULES},
                    rules={rule: st.per_history(*outcomes[rule]) for rule in RULES},
                    paired={rule: dict(completion=float((outcomes[rule][1].mean(0) - fixed).mean())) for rule in ADAPTIVE})
        budgets[str(n)] = block
    return dict(r=r, scenario=scenario, budgets=budgets)


def _job(job):
    return run_split(*job)


def summarize(repeats):
    out = {}
    for scenario in SCENARIOS:
        out[scenario] = {}
        for n in (str(b) for b in BUDGETS):
            out[scenario][n] = {}
            for info, sel in product(INFOS, SELECTIONS):
                rows = [repeats[str(r)][scenario]['budgets'][n][info][sel] for r in range(st.SPLITS)]
                out[scenario][n].setdefault(info, {})[sel] = dict(
                    meets_per_history={rule: int(sum(all(row['rules'][rule][h]['success_rate'] >= br.Q
                                                         for h in br.HISTORIES) for row in rows)) for rule in RULES},
                    certified_both={rule: (int(sum(all(row['choices'][rule][h]['certified'] for h in br.HISTORIES)
                                                   for row in rows)) if sel == 'certified' else None) for rule in RULES},
                    fallback_any={rule: int(sum(any(row['choices'][rule][h]['fallback'] for h in br.HISTORIES)
                                                for row in rows)) for rule in RULES},
                    success={rule: {h: st.stats([row['rules'][rule][h]['success_rate'] for row in rows])
                                    for h in br.HISTORIES} for rule in RULES},
                    completion={rule: st.stats([row['rules'][rule]['all']['mean_completion_min'] for row in rows])
                                for rule in RULES},
                    paired_completion={rule: st.stats([row['paired'][rule]['completion'] for row in rows])
                                       for rule in ADAPTIVE},
                    faster={rule: int(sum(row['paired'][rule]['completion'] < 0 for row in rows)) for rule in ADAPTIVE})
            retained = {}
            for sel in SELECTIONS:
                retained[sel] = {}
                for rule in ADAPTIVE:
                    gain = {info: float(np.median([-repeats[str(r)][scenario]['budgets'][n][info][sel]['paired'][rule]['completion']
                                                   for r in range(st.SPLITS)])) for info in INFOS}
                    retained[sel][rule] = gain['one_probe'] / gain['full_curve'] if gain['full_curve'] > 0 else None
            out[scenario][n]['retained_gain'] = retained
    return out


def statements(summary):
    e1 = json.loads(st.RESULTS.read_text(encoding='utf-8'))['summary']['0.005']['certified']
    budgets = [str(b) for b in BUDGETS]
    return {
        'E2-S1': {n: {sel: {rule: dict(splits_meeting=summary['baseline'][n]['one_probe'][sel]['meets_per_history'][rule],
                                       median_completion=summary['baseline'][n]['one_probe'][sel]['completion'][rule]['median'])
                            for rule in RULES} for sel in SELECTIONS} for n in budgets},
        'E2-S2': dict(retained_gain={n: summary['baseline'][n]['retained_gain'] for n in budgets},
                      e1_full_information_certified=dict(paired_completion_median={k: v['median'] for k, v in e1['paired_completion'].items()},
                                                         completion_median={k: v['median'] for k, v in e1['completion'].items()})),
        'E2-S3': {n: {sel: {rule: summary['hidden_lag_8'][n]['one_probe'][sel]['meets_per_history'][rule] for rule in RULES}
                      for sel in SELECTIONS} for n in budgets}}


def main():
    load_protocol()
    jobs = [(r, scenario) for r in range(st.SPLITS) for scenario in SCENARIOS]
    with ProcessPoolExecutor(max_workers=min(20, os.cpu_count() or 1)) as pool:
        results = list(pool.map(_job, jobs))
    repeats = {}
    for (r, scenario), result in zip(jobs, results):
        repeats.setdefault(str(r), {})[scenario] = result
    summary = summarize(repeats)
    result = dict(status='complete', protocol_sha256=protocol_sha256(),
                  frozen_utc=json.loads(FREEZE.read_text(encoding='utf-8'))['frozen_utc'],
                  repeats=repeats, summary=summary, statements=statements(summary),
                  limitations=['Simulation of one published model structure; not evidence about measured cells.',
                               'Seen posterior vectors with new roles and noise; uniform rung assignment only.',
                               'The hidden lag is a stylized stress test.'])
    RESULTS.write_text(json.dumps(result, indent=2) + '\n', newline='\n')
    for scenario in SCENARIOS:
        for n in (str(b) for b in BUDGETS):
            for info, sel in product(INFOS, SELECTIONS):
                b = summary[scenario][n][info][sel]
                print(scenario, n, info, sel, 'meets', b['meets_per_history'], 'fallback', b['fallback_any'],
                      'T', {k: round(v['median'], 1) for k, v in b['completion'].items()},
                      'dT', {k: round(v['median'], 2) for k, v in b['paired_completion'].items()})
            print(scenario, n, 'retained', summary[scenario][n]['retained_gain'])


if __name__ == '__main__':
    main()
