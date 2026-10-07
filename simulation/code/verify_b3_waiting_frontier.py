"""Audit the per-history calibration, oracle-gap and frontier study.

Checks the vectorized crossing against the version-2 rule, that per-history
fixed waits at target 0.90 equal the frozen per-history conditioned waits,
that choices ignore evaluation labels, the oracle against readiness version 2,
the gap-closure arithmetic, and recomputes one split from scratch.
"""
import json
import sys

import numpy as np

import b3_readiness as br
import b3_waiting as bw
import b3_waiting_baselines as bl
import b3_waiting_frontier as wf
import b3_waiting_robustness as rb

VERIFY = br.OUT / 'waiting_frontier_verification_results.json'


def crossing_check():
    rng = np.random.default_rng(1)
    probability = rng.random((2, 7, br.GRID.size)) ** 3
    k = wf.crossing_index(probability, wf.THRESHOLDS)
    for i, c in enumerate(wf.THRESHOLDS):
        expected = np.apply_along_axis(br.first_crossing, -1, probability, c)
        assert (k[i] == expected).all(), c
    return dict(thresholds=len(wf.THRESHOLDS), matches_first_crossing=True)


def fixed_check(tables):
    success = br.success_table(tables)
    cal, _ = bw.split_samples(success.shape[1])
    options = wf.candidates({name: np.zeros((2, cal.size, br.GRID.size)) for name in
                             ('current', 'history') + tuple(f'smoothed_{t:g}' for t in bw.SMOOTH_TAUS)},
                            success, cal, br.DEADLINE_MIN)['fixed']
    waits = [wf.choose(options, h, br.Q)['wait_min'] for h in range(2)]
    frozen = bl.conditioned_fixed(success, cal, per_history=True)['waits_min']
    assert waits == frozen, (waits, frozen)
    return dict(per_history_waits=waits, equals_frozen_conditioned_per_history=True)


def leakage_check(tables, noise=.005):
    success = br.success_table(tables)
    cal, ev = bw.split_samples(success.shape[1])
    observed = bw.noisy_observations(tables, noise)
    flipped = success.copy()
    flipped[:, ev, :] = ~flipped[:, ev, :]
    choices = []
    for table in (success, flipped):
        probability = bw.estimator_probabilities(tables, table, noise, cal, cal, observed)
        options = wf.candidates(probability, table, cal, br.DEADLINE_MIN)
        choices.append({n: [wf.choose(options[n], h, .92) for h in range(2)] for n in wf.RULES})
    assert choices[0] == choices[1]
    return dict(evaluation_labels_flipped=True, choices_unchanged=True)


def oracle_check(saved):
    v2 = json.loads(br.RESULTS.read_text(encoding='utf-8'))['policies']['oracle']
    k = saved['kappa_feasibility']['0.50']['oracle']['all']
    assert abs(k['success_rate'] - v2['success_rate']) < 1e-12 and abs(k['mean_completion_min'] - v2['mean_completion_min']) < 1e-9
    return dict(kappa_0_5_oracle_equals_readiness_v2=True)


def gap_check(saved):
    worst = 0.
    for r, by_noise in saved['repeats'].items():
        for noise, row in by_noise.items():
            for target, cell in row['cells'].items():
                expected = wf.gap_closure(cell['rules'])
                for name, parts in expected.items():
                    for part, value in parts.items():
                        worst = max(worst, abs(value - cell['gap_closure'][name][part]))
    v3 = json.loads(bw.RESULTS.read_text(encoding='utf-8'))['noise']['0.005']['evaluation']
    conditioned = json.loads(bl.RESULTS.read_text(encoding='utf-8'))['primary']['conditioned']['pooled']['evaluation']['all']
    expected = ((conditioned['mean_completion_min'] - v3['history']['mean_completion_min'])
                / (conditioned['mean_completion_min'] - v3['oracle']['mean_completion_min']))
    worst = max(worst, abs(expected - saved['version3_gap_closure']['primary']['history']['vs_conditioned']))
    assert worst < 1e-12, worst
    return dict(max_abs_difference=worst)


def recompute_check(saved, r=3, noise=.01):
    fresh = wf.run_repeat(r, noise)
    stored = saved['repeats'][str(r)][f'{noise:g}']
    worst = 0.
    for target, cell in fresh['cells'].items():
        assert cell['choices'] == stored['cells'][target]['choices'], target
        for name, parts in cell['rules'].items():
            for part, row in parts.items():
                for key, value in row.items():
                    worst = max(worst, abs(value - stored['cells'][target]['rules'][name][part][key]))
    assert worst < 1e-12, worst
    return dict(repeat=r, noise=noise, max_abs_difference=worst)


def main():
    wf.load_protocol()
    tables = dict(np.load(br.TABLES))
    saved = json.loads(wf.RESULTS.read_text(encoding='utf-8'))
    checks = dict(crossing=crossing_check(), fixed=fixed_check(tables), leakage=leakage_check(tables),
                  oracle=oracle_check(saved), gap=gap_check(saved), recompute=recompute_check(saved))
    result = dict(status='passed', protocol_sha256=wf.protocol_sha256(), checks=checks)
    VERIFY.write_text(json.dumps(result, indent=2) + '\n', newline='\n')
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
