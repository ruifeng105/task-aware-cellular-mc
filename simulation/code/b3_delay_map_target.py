"""Condition map for model-based waiting on B3 at calibration targets 0.92 and 0.94.

Implements configs/b3_delay_map_target_protocol.json, frozen before its only
run. Repeats b3_delay_map with the per-history calibration target raised from
0.90; the rule options of each split are computed once and scored at all
targets, and the 0.90 cells must reproduce the frozen map. Adds a per-history
class and calibration-fallback counts. Simulation of one published model
structure in which the B3 rules share the plant structure.
"""
import hashlib
import json
import os

import numpy as np

import b3_delay_map as dm
import b3_readiness as br
import b3_waiting as bw
import b3_waiting_frontier as wf
import b3_waiting_robustness as rb
from parallel_jobs import run_jobs

PROTOCOL = br.ROOT / 'configs/b3_delay_map_target_protocol.json'
FREEZE = br.OUT / 'b3_delay_map_target_protocol_freeze.json'
RESULTS = br.OUT / 'delay_map_target_results.json'
TARGETS = (.90, .92, .94)
REPORTED = ('0.92', '0.94')
COMPARED = ('b3_predictive', 'smoothed_slope')


def protocol_sha256():
    return hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()


def load_protocol():
    if protocol_sha256() != json.loads(FREEZE.read_text(encoding='utf-8'))['protocol_sha256']:
        raise SystemExit('Protocol differs from its freeze record; refusing to run.')
    dm.load_protocol()
    protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    for rel, expected in protocol['inputs_sha256'].items():
        if hashlib.sha256((br.ROOT.parent / rel).read_bytes()).hexdigest() != expected:
            raise SystemExit(f'Input {rel} differs from the hash pinned in the protocol.')
    return protocol


def evaluate_at(options_cal, options_ev, oracle, target):
    """b3_delay_map.evaluate_cell with the calibration target as a parameter."""
    choices = {name: {hist: wf.choose(options_cal[name], h, target) for h, hist in enumerate(br.HISTORIES)}
               for name in dm.RULES}
    outcomes = {}
    for name in dm.RULES:
        parts = [wf.lookup(options_ev[name], choices[name][hist]) for hist in br.HISTORIES]
        outcomes[name] = (np.stack([parts[h][0][h] for h in range(len(parts))]),
                          np.stack([parts[h][1][h] for h in range(len(parts))]))
    rules = {name: rb.summarise(*pair) for name, pair in outcomes.items()}
    rules['oracle'] = rb.summarise(*oracle)
    per_sample = {name: (ok.mean(0), completion.mean(0)) for name, (ok, completion) in outcomes.items()}
    paired = {f'{a}-{b}': dict(success=float((per_sample[a][0] - per_sample[b][0]).mean()),
                               completion=float((per_sample[a][1] - per_sample[b][1]).mean())) for a, b in dm.PAIRS}
    return dict(choices=choices, rules=rules, paired=paired,
                meets_per_history={name: bool(all(rules[name][h]['success_rate'] >= br.Q for h in br.HISTORIES))
                                   for name in dm.RULES})


def job_parts(r, noise, interval):
    tables = dict(np.load(br.TABLES))
    success = br.success_table(tables)
    split_seed, noise_seed = rb.seeds(r)
    cal, ev = bw.split_samples(success.shape[1], seed=split_seed)
    observed = bw.noisy_observations(tables, noise, seed=noise_seed)
    return tables, success, cal, ev, observed


def run_job(r, noise, interval):
    tables, success, cal, ev, observed = job_parts(r, noise, interval)
    options_cal = dm.rule_options(tables, success, observed, noise, interval, cal, cal)
    options_ev = dm.rule_options(tables, success, observed, noise, interval, cal, ev)
    oracle = wf.oracle_outcomes(success, ev, br.DEADLINE_MIN)
    cells = {f'{t:.2f}': {f'{d}': evaluate_at(options_cal[d], options_ev[d], oracle, t) for d in dm.DELAYS} for t in TARGETS}
    return json.loads(json.dumps(dict(r=r, noise=noise, interval=interval, cells=cells)))


def _job(job):
    return run_job(*job)


def summarize(rows):
    """b3_delay_map.summarize plus per-history success, calibration counts and the per-history class."""
    out = dm.summarize(rows)
    reached = {name: int(sum(all(row['choices'][name][h]['target_reached'] for h in br.HISTORIES) for row in rows))
               for name in dm.RULES}
    out['target_reached_on_calibration'] = reached
    if min(reached[name] for name in COMPARED) < dm.HELPS_SPLITS:
        per_history = 'not classifiable'
    elif out['map_class'] == 'model helps' and out['meets_per_history']['b3_predictive'] >= dm.HELPS_SPLITS:
        per_history = 'model helps'
    else:
        per_history = 'reporter suffices'
    out['per_history_class'] = per_history
    out['success_gap_median'] = (out['success']['b3_predictive']['all']['median']
                                 - out['success']['smoothed_slope']['all']['median'])
    return out


def reproduction_difference(splits_090):
    """Largest difference between the 0.90 cells and the frozen delay map (choices must match exactly)."""
    frozen = json.loads(dm.RESULTS.read_text(encoding='utf-8'))
    worst = 0.
    for key, rows in splits_090.items():
        for r, cell in rows.items():
            old = frozen['splits'][key][r]
            assert cell['choices'] == old['choices'], (key, r)
            for rule, parts in cell['rules'].items():
                for part, row in parts.items():
                    worst = max(worst, *(abs(v - old['rules'][rule][part][k]) for k, v in row.items()))
            worst = max(worst, *(abs(v[k] - old['paired'][p][k]) for p, v in cell['paired'].items() for k in v))
        assert dm.summarize([rows[str(i)] for i in range(rb.REPEATS)])['map_class'] == frozen['summary'][key]['map_class'], key
    return worst


def class_changes(summary, splits_090):
    frozen = json.loads(dm.RESULTS.read_text(encoding='utf-8'))['summary']
    out = {}
    for key, block in summary['0.94'].items():
        if block['map_class'] != frozen[key]['map_class']:
            old = summarize([splits_090[key][str(r)] for r in range(rb.REPEATS)])
            out[key] = dict(class_090=frozen[key]['map_class'], class_094=block['map_class'],
                            faster_090=frozen[key]['b3_faster_than_smoothed_slope'], faster_094=block['b3_faster_than_smoothed_slope'],
                            success_gap_090=old['success_gap_median'], success_gap_094=block['success_gap_median'])
    return out


def headline_cells(summary):
    frozen = json.loads(dm.RESULTS.read_text(encoding='utf-8'))['summary']
    return sorted(key for key, block in summary['0.94'].items()
                  if frozen[key]['map_class'] == 'model helps' and block['per_history_class'] == 'model helps')


def main():
    load_protocol()
    jobs = [(r, noise, interval) for noise in dm.NOISES for interval in dm.INTERVALS for r in range(rb.REPEATS)]
    workers = int(os.environ.get('B3_WORKERS', min(32, os.cpu_count() or 1)))   # runtime only, not part of the protocol
    results = run_jobs(_job, jobs, workers)
    splits = {f'{t:.2f}': {} for t in TARGETS}
    for (r, noise, interval), result in zip(jobs, results):
        for t, cells in result['cells'].items():
            for d, cell in cells.items():
                splits[t].setdefault(dm.cell_key(noise, interval, int(d)), {})[str(r)] = cell
    worst = reproduction_difference(splits['0.90'])
    assert worst < 1e-12, worst
    summary = {t: {key: summarize([rows[str(r)] for r in range(rb.REPEATS)]) for key, rows in splits[t].items()} for t in REPORTED}
    result = dict(status='complete', protocol_sha256=protocol_sha256(),
                  frozen_utc=json.loads(FREEZE.read_text(encoding='utf-8'))['frozen_utc'],
                  reproduction_at_090_max_abs_difference=worst,
                  grid=dict(delays_min=list(dm.DELAYS), intervals_min=list(dm.INTERVALS), noise_sd=list(dm.NOISES)),
                  splits={t: splits[t] for t in REPORTED}, summary=summary,
                  class_changes_090_to_094=class_changes(summary, splits['0.90']),
                  headline_model_helps_cells=headline_cells(summary),
                  limitations=['Simulation of one published model structure; the B3 rules share the plant structure and the model is correct.',
                               'Target 0.94 was chosen after the frontier study had been evaluated.',
                               'Repeats redraw the split and observation noise only; one finite posterior set is reused.'])
    RESULTS.write_text(json.dumps(result, indent=2) + '\n', newline='\n')
    for key, block in summary['0.94'].items():
        print(key, block['map_class'], block['per_history_class'], round(block['map_value_min'], 2),
              block['b3_faster_than_smoothed_slope'], 'met', block['meets_per_history'],
              'reached', block['target_reached_on_calibration'])
    print('headline cells', result['headline_model_helps_cells'])
    print('class changes', json.dumps(result['class_changes_090_to_094']))


if __name__ == '__main__':
    main()
