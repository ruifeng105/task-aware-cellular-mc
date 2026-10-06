"""Audit the history-conditioned fixed waits and the fine threshold grid.

Checks the joint selection on a synthetic case where the pooled and the
per-history constraints differ, that the selection ignores evaluation
plants, recomputes the primary-split evaluation from the saved waits, and
recomputes one point of the fine-grid frontier.
"""
import json
import sys

import numpy as np

import b3_readiness as br
import b3_waiting as bw
import b3_waiting_baselines as bl


def synthetic_check():
    grid = np.array([0., 10., 100.])
    success = np.zeros((2, 10, 3), dtype=bool)
    success[0] = True
    success[1, :8, 1] = True
    success[1, :, 2] = True
    plants = np.arange(10)
    pooled = bl.conditioned_fixed(success, plants, per_history=False, grid=grid)
    strict = bl.conditioned_fixed(success, plants, per_history=True, grid=grid)
    assert pooled['waits_min'] == [0., 10.] and strict['waits_min'] == [0., 100.], (pooled, strict)
    assert abs(pooled['calibration']['mean_completion_min'] - 36.) < 1e-12
    none = bl.conditioned_fixed(np.zeros((2, 10, 3), dtype=bool), plants, grid=grid)
    assert none['target_reached'] is False and none['waits_min'] == [100., 100.]
    return dict(pooled=pooled['waits_min'], per_history=strict['waits_min'], fallback=True)


def leakage_check(tables, saved):
    success = br.success_table(tables)
    cal, ev = bw.split_samples(success.shape[1])
    flipped = success.copy()
    flipped[:, ev, :] = ~flipped[:, ev, :]
    for rule in ('pooled', 'per_history'):
        choice = bl.conditioned_fixed(flipped, cal, per_history=rule == 'per_history')
        assert choice['waits_min'] == saved['primary']['conditioned'][rule]['waits_min'], rule
    return dict(evaluation_labels_flipped=True, choices_unchanged=True)


def recompute_check(tables, saved):
    success = br.success_table(tables)
    _, ev = bw.split_samples(success.shape[1])
    worst = 0.
    for rule, block in saved['primary']['conditioned'].items():
        fresh = bl.evaluate_fixed(success, ev, block['waits_min'])
        for part, row in block['evaluation'].items():
            for key, value in row.items():
                worst = max(worst, abs(fresh[part][key] - value))
    assert worst < 1e-12, worst
    return dict(max_abs_difference=worst)


def frontier_check(tables, saved, noise=.01, threshold=.93):
    success = br.success_table(tables)
    cal, ev = bw.split_samples(success.shape[1])
    probability = bw.estimator_probabilities(tables, success, noise, cal, ev)
    fresh = bw.outcomes(probability['history'], success, ev, threshold)
    row = next(p for p in saved['fine_grid'][f'{noise:g}']['frontier']['history'] if abs(p['threshold'] - threshold) < 1e-9)
    worst = max(abs(fresh[k] - row[k]) for k in ('success_rate', 'mean_completion_min'))
    assert worst < 1e-12, worst
    return dict(noise=noise, threshold=threshold, max_abs_difference=worst)


def main():
    bl.load_protocol()
    tables = dict(np.load(br.TABLES))
    saved = json.loads(bl.RESULTS.read_text(encoding='utf-8'))
    report = dict(status='passed', protocol_sha256=bl.protocol_sha256(), synthetic=synthetic_check(),
                  leakage=leakage_check(tables, saved), recomputation=recompute_check(tables, saved),
                  frontier=frontier_check(tables, saved),
                  scope='Simulation of the published B3 model; not evidence about measured cells.')
    (br.OUT / 'waiting_baselines_verification_results.json').write_text(json.dumps(report, indent=2) + '\n', newline='\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    sys.exit(main())
