"""Audit of E2, calibration from one probe per calibration receiver (configs/b3_waiting_budget_protocol.json).

Checks: freeze hash and pinned inputs; the label accessor refuses a second request and was called exactly n times
per rule and history; readiness is non-decreasing in time and decision indices are non-increasing along every
ladder (the conditions of the conservative one-probe test); pool-adjacent-violators equals a brute-force isotonic
fit; a hand-computed certified selection; leakage (inverted test outcomes leave every choice unchanged); split 0
recomputed from scratch; summary and statements re-derived with independent code.
Usage: python verify_b3_waiting_budget.py
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')  # one BLAS thread per process: no oversubscription in pools
import hashlib
import json
from itertools import product

import numpy as np

import b3_readiness as br
import b3_waiting as bw
import b3_waiting_budget as bb
import b3_waiting_strict as st
import strict_eval as se
from verify_b3_waiting_strict import close

OUT = br.OUT / 'waiting_budget_verification_results.json'


def freeze_check(protocol):
    freeze = json.loads(bb.FREEZE.read_text(encoding='utf-8'))
    assert hashlib.sha256(bb.PROTOCOL.read_bytes()).hexdigest() == freeze['protocol_sha256']
    for path, digest in protocol['inputs_sha256'].items():
        assert hashlib.sha256((br.ROOT.parent / path).read_bytes()).hexdigest() == digest, path
    return dict(protocol_sha256=freeze['protocol_sha256'])


def oracle_check(result):
    success = np.zeros((2, 5, 61), bool)
    oracle = bb.LabelOracle(success, np.arange(5))
    oracle.probe(0, 1, 3)
    try:
        oracle.probe(0, 1, 7)
    except RuntimeError:
        pass
    else:
        raise AssertionError('second probe of one receiver was allowed')
    assert oracle.calls == 1
    for r, block in result['repeats'].items():
        for scenario, run in block.items():
            for n, cells in run['budgets'].items():
                assert cells['probe_calls'] == {rule: {h: int(n) for h in br.HISTORIES} for rule in bb.RULES}, (r, scenario, n)
    return dict(second_probe_refused=True, calls_equal_budget=True)


def monotone_check(tables):
    model = br.success_table(tables)
    for scenario in bb.SCENARIOS:
        plant = bb.plant_success(model, scenario).astype(int)
        assert (np.diff(plant, axis=-1) >= 0).all(), scenario
    R, C, _ = st.roles(0)
    observed = bw.noisy_observations(tables, bb.NOISE, seed=st.noise_seed(0))
    probability = bw.estimator_probabilities(tables, model, bb.NOISE, R, C, observed)
    for rule in bb.RULES:
        k = bb.ladder_indices(probability, rule, 2, len(C))
        assert k.shape == (bb.RUNGS, 2, len(C)) and (np.diff(k, axis=0) <= 0).all(), rule
    return dict(readiness_non_decreasing=True, ladders_non_increasing=True)


def brute_isotonic(values, weights):
    """Non-increasing isotonic fit: f_i = min_{j<=i} max_{k>=i} weighted mean of values[j..k]."""
    n = len(values)
    mean = lambda j, k: float(np.dot(values[j:k + 1], weights[j:k + 1]) / weights[j:k + 1].sum())
    return np.array([min(max(mean(j, k) for k in range(i, n)) for j in range(i + 1)) for i in range(n)])


def pava_check():
    rng = np.random.default_rng(5)
    for _ in range(200):
        n = int(rng.integers(1, 9))
        values, weights = rng.random(n), rng.integers(1, 5, n).astype(float)
        assert np.allclose(bb.pava_nonincreasing(values, weights), brute_isotonic(values, weights))
    return dict(random_cases=200)


def selection_check():
    # 100 receivers at rungs 0-3, three failures at rung 3. Rung 0 is supported by all 100 (97 successes,
    # p = 0.0078), rung 1 by the 70 at rungs 1-3 (67 successes, p = 0.071 > 0.025): the sequence stops after rung 0.
    positions = np.array([0] * 30 + [1] * 30 + [2] * 20 + [3] * 20)
    labels = np.ones(100, bool)
    labels[-3:] = False
    expected = None
    for j in range(bb.RUNGS):
        mask = positions >= j
        if not mask.any() or se.binom_pvalue(int(labels[mask].sum()), int(mask.sum()), .9) > bb.ALPHA:
            break
        expected = j
    assert expected == 0 and bb.certified_one_probe(positions, labels) == 0
    assert bb.certified_one_probe(positions, np.zeros(100, bool)) is None
    # full curves: 40/40 successes at rungs 0-4 (p = 0.0148), 34/40 from rung 5 on
    ok = np.ones((bb.RUNGS, 40), bool)
    ok[5:, :6] = False
    assert bb.certified_full(ok) == 4 and bb.plugin_full(ok) == 4
    # plug-in from one probe: rung means 1, 1, 1, 0.85 are already non-increasing; the most aggressive >= 0.9 is 2
    assert bb.plugin_one_probe(positions, labels) == 2
    return dict(hand_computed_certified_rung=expected)


def leakage_check(tables, saved):
    model = br.success_table(tables)
    _, _, T = st.roles(0)
    for scenario in bb.SCENARIOS:
        plant = bb.plant_success(model, scenario).copy()
        plant[:, T, :] = ~plant[:, T, :]
        rerun = bb.run_split(0, scenario, tables=tables, plant=plant)
        choices = lambda run: {n: {i: {s: c['choices'] for s, c in cells[i].items()} for i in bb.INFOS}
                               for n, cells in run['budgets'].items()}
        close(choices(rerun), choices(saved[scenario]))
    return dict(test_outcomes_inverted_choices_unchanged=True)


def summary_check(result):
    repeats, summary = result['repeats'], result['summary']
    for scenario, n, info, sel in product(bb.SCENARIOS, (str(b) for b in bb.BUDGETS), bb.INFOS, bb.SELECTIONS):
        rows = [repeats[str(r)][scenario]['budgets'][n][info][sel] for r in range(st.SPLITS)]
        block = summary[scenario][n][info][sel]
        for rule in bb.RULES:
            meets = sum(all(row['rules'][rule][h]['success_rate'] >= .9 for h in br.HISTORIES) for row in rows)
            assert block['meets_per_history'][rule] == meets
            comp = float(np.median([row['rules'][rule]['all']['mean_completion_min'] for row in rows]))
            assert abs(block['completion'][rule]['median'] - comp) < 1e-12
        for rule in bb.ADAPTIVE:
            diffs = [row['paired'][rule]['completion'] for row in rows]
            assert block['faster'][rule] == sum(d < 0 for d in diffs)
    for scenario, n, sel in product(bb.SCENARIOS, (str(b) for b in bb.BUDGETS), bb.SELECTIONS):
        for rule in bb.ADAPTIVE:
            gains = {info: float(np.median([-repeats[str(r)][scenario]['budgets'][n][info][sel]['paired'][rule]['completion']
                                            for r in range(st.SPLITS)])) for info in bb.INFOS}
            expected = gains['one_probe'] / gains['full_curve'] if gains['full_curve'] > 0 else None
            got = summary[scenario][n]['retained_gain'][sel][rule]
            assert (got is None and expected is None) or abs(got - expected) < 1e-12, (scenario, n, sel, rule)
    s = result['statements']
    for n, sel, rule in product((str(b) for b in bb.BUDGETS), bb.SELECTIONS, bb.RULES):
        assert s['E2-S1'][n][sel][rule]['splits_meeting'] == summary['baseline'][n]['one_probe'][sel]['meets_per_history'][rule]
        assert s['E2-S3'][n][sel][rule] == summary['hidden_lag_8'][n]['one_probe'][sel]['meets_per_history'][rule]
    return dict(summary_rederived=True, statements_rederived=True)


def main():
    protocol = bb.load_protocol()
    result = json.loads(bb.RESULTS.read_text(encoding='utf-8'))
    assert result['status'] == 'complete' and result['protocol_sha256'] == hashlib.sha256(bb.PROTOCOL.read_bytes()).hexdigest()
    tables = dict(np.load(br.TABLES))
    checks = dict(freeze=freeze_check(protocol), oracle=oracle_check(result), monotone=monotone_check(tables),
                  pava=pava_check(), selection=selection_check(),
                  leakage=leakage_check(tables, result['repeats']['0']), summary=summary_check(result))
    for scenario in bb.SCENARIOS:
        close(bb.run_split(0, scenario, tables=tables), result['repeats']['0'][scenario], f'split 0 {scenario}')
    checks['recompute'] = dict(recomputed_split=0)
    OUT.write_text(json.dumps(dict(status='passed', checks=checks), indent=2) + '\n', newline='\n')
    print('E2 one-probe calibration verification: passed')


if __name__ == '__main__':
    main()
