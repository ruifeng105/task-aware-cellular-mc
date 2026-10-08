"""Audit of the sample roles in the earlier waiting studies (revision P0).

The author posterior file repeats some parameter vectors (resampling), so leaving a plant out by index does not
remove an identical twin from its references. This audit counts such twins in the 20 calibration/evaluation
splits of b3_waiting_frontier, re-scores that study's evaluation (noise 0.005; targets 0.90 and 0.94; its saved
choices) with and without the evaluation plants that have a twin among the references, summarizes how often a
calibration target was not reached and what the rule then did, and lists how the measured natural probes differ
from the simulated task. Descriptive only; deterministic.
Usage: python audit_waiting_design.py
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')  # one BLAS thread per process: no oversubscription in pools
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np

import b3_model as b3
import b3_readiness as br
import b3_waiting as bw
import b3_waiting_frontier as fr
import b3_waiting_robustness as rb

OUT = br.ROOT / 'results/audit/waiting_design_audit.json'
NOISE, TARGETS = .005, ('0.90', '0.94')


def twin_groups(posterior):
    """List of index arrays, one per distinct parameter vector."""
    _, inverse = np.unique(posterior, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    return [np.flatnonzero(inverse == g) for g in range(inverse.max() + 1)]


def twin_of(groups, n):
    twins = [set() for _ in range(n)]
    for group in groups:
        for i in group:
            twins[i] = set(group.tolist()) - {int(i)}
    return twins


def rescore(r):
    tables = dict(np.load(br.TABLES))
    success = br.success_table(tables)
    twins = twin_of(twin_groups(b3.load_posterior()), success.shape[1])
    split_seed, noise_seed = rb.seeds(r)
    cal, ev = bw.split_samples(success.shape[1], seed=split_seed)
    cal_set = set(cal.tolist())
    observed = bw.noisy_observations(tables, NOISE, seed=noise_seed)
    on_ev = bw.estimator_probabilities(tables, success, NOISE, cal, ev, observed)
    options_ev = fr.candidates(on_ev, success, ev, br.DEADLINE_MIN)
    saved = json.loads(fr.RESULTS.read_text(encoding='utf-8'))['repeats'][str(r)]['0.005']['cells']
    affected = np.array([bool(twins[p] & cal_set) for p in ev])
    out = dict(r=r, evaluation_with_twin_in_references=int(affected.sum()),
               calibration_with_twin_in_loo_references=int(sum(bool(twins[p] & (cal_set - {int(p)})) for p in cal)),
               targets={})
    for target in TARGETS:
        outcomes = fr.evaluated(options_ev, saved[target]['choices'])
        block = {}
        for name, (ok, completion) in outcomes.items():
            stored = saved[target]['rules'][name]
            assert all(abs(float(ok[h].mean()) - stored[hist]['success_rate']) < 1e-12
                       for h, hist in enumerate(br.HISTORIES)), (r, target, name)
            block[name] = {label: {hist: dict(success_rate=float(ok[h][mask].mean()),
                                              mean_completion_min=float(completion[h][mask].mean()))
                                   for h, hist in enumerate(br.HISTORIES)}
                           for label, mask in (('all', np.ones_like(affected)), ('without_twin', ~affected),
                                               ('twin_only', affected)) if mask.any()}
        fixed = outcomes['fixed'][1].mean(0)
        block['paired_completion_minus_fixed'] = {
            name: {label: float((outcomes[name][1].mean(0) - fixed)[mask].mean())
                   for label, mask in (('all', np.ones_like(affected)), ('without_twin', ~affected),
                                       ('twin_only', affected)) if mask.any()}
            for name in fr.ADAPTIVE}
        out['targets'][target] = block
    return out


def infeasibility():
    summary = {}
    repeats = json.loads(fr.RESULTS.read_text(encoding='utf-8'))['repeats']
    for noise in ('0.005', '0.01'):
        for target in ('0.90', '0.92', '0.94'):
            for name in fr.RULES:
                for hist in br.HISTORIES:
                    missed = [r for r in range(rb.REPEATS)
                              if not repeats[str(r)][noise]['cells'][target]['choices'][name][hist]['target_reached']]
                    summary[f'{noise}/{target}/{name}/{hist}'] = len(missed)
    return dict(target_not_reached_splits=summary,
                fallback='b3_waiting_frontier.choose: if no parameter reaches the target on calibration, the parameter '
                         'with the highest calibration success is used (ties: smaller completion), the probe is still '
                         'sent and the split is scored like any other; target_reached is recorded per rule and history.')


TASK_DIFFERENCES = [
    dict(item='probe command', measured_3_20='3-min pulse at 25 ng/ml, 20 min after the previous 3-min pulse',
         measured_mixed='5-min pulse at 25 ng/ml, 60 min after a 30-min pulse', simulation='5-min pulse at 25 ng/ml'),
    dict(item='naive denominator', measured_3_20="the same cell's rise to the first 3-min pulse",
         measured_mixed='median rise of other cells to a single 5-min pulse (fgf_sp_5)',
         simulation="the same posterior sample's rise to a 5-min pulse on an unstimulated receiver"),
    dict(item='response', measured_3_20='noisy reporter, 3-frame moving average, 2-min frames; rise above the mean '
                                        'of the two frames before the probe',
         measured_mixed='as for 3/20', simulation='noise-free model reporter on a 1-min grid; rise above F(w)'),
    dict(item='window', measured_3_20='20 min', measured_mixed='20 min', simulation='20 min'),
    dict(item='role', measured_3_20='model check (fixed waits, no decision)',
         measured_mixed='model check (one fixed wait, no decision)',
         simulation='the task of Eq. (2) used by every waiting rule'),
]


def main():
    posterior = b3.load_posterior()
    groups = twin_groups(posterior)
    with ProcessPoolExecutor(max_workers=min(20, os.cpu_count() or 1)) as pool:
        splits = list(pool.map(rescore, range(rb.REPEATS)))
    summary = {}
    for target in TARGETS:
        summary[target] = {}
        for name in fr.ADAPTIVE:
            values = {label: [s['targets'][target]['paired_completion_minus_fixed'][name][label] for s in splits]
                      for label in ('all', 'without_twin', 'twin_only')}
            summary[target][name] = {label: rb.stats(v) for label, v in values.items()}
        summary[target]['success_without_twin'] = {
            name: {hist: rb.stats([s['targets'][target][name]['without_twin'][hist]['success_rate'] for s in splits])
                   for hist in br.HISTORIES} for name in fr.RULES}
        summary[target]['meets_per_history_without_twin'] = {
            name: int(sum(all(s['targets'][target][name]['without_twin'][h]['success_rate'] >= br.Q
                              for h in br.HISTORIES) for s in splits)) for name in fr.RULES}
    result = dict(
        status='complete',
        posterior=dict(rows=int(posterior.shape[0]), distinct_vectors=len(groups),
                       multiplicity=dict(sorted(Counter(len(g) for g in groups).items()))),
        twins_per_split=dict(evaluation_with_twin_in_references=[s['evaluation_with_twin_in_references'] for s in splits],
                             calibration_with_twin_in_loo_references=[s['calibration_with_twin_in_loo_references']
                                                                      for s in splits]),
        rescored=summary, splits=splits, infeasibility=infeasibility(), task_differences=TASK_DIFFERENCES,
        reading='Rules were chosen with the twins present, so the without-twin rows re-score the same choices on a '
                'subset; they bound the size of the self-match effect but are not a deduplicated rerun.')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + '\n', newline='\n')
    print('distinct', len(groups), result['posterior']['multiplicity'])
    print('eval twins', result['twins_per_split']['evaluation_with_twin_in_references'])
    print('cal twins', result['twins_per_split']['calibration_with_twin_in_loo_references'])
    for target in TARGETS:
        print(target, {n: {k: round(v['median'], 3) for k, v in summary[target][n].items()} for n in fr.ADAPTIVE})
        print(target, 'meets without twin', summary[target]['meets_per_history_without_twin'])
    print('infeasible', {k: v for k, v in result['infeasibility']['target_not_reached_splits'].items() if v})


if __name__ == '__main__':
    main()
