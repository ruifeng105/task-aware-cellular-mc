"""Audit the waiting robustness study without changing any setting.

Checks that repeat 0 reproduces version 3, that repeat seeds and splits are
distinct, the objective-aligned threshold rule on a case where the first
feasible threshold is not the fastest, that aligned choices ignore evaluation
plants, and recomputes one further repeat.
"""
import json
import sys

import numpy as np

import b3_readiness as br
import b3_waiting as bw
import b3_waiting_robustness as rb


def version3_check(saved):
    v3 = json.loads(bw.RESULTS.read_text(encoding='utf-8'))['noise']
    names = {'current': 'current', 'smoothed': 'smoothed', 'history': 'history', 'fixed': 'fixed_calibrated', 'oracle': 'oracle'}
    worst = 0.
    for noise, block in v3.items():
        r0 = saved['repeats']['0'][noise]
        assert r0['choices'] == block['choices'], noise
        for ours, theirs in names.items():
            for key in ('success_rate', 'mean_completion_min'):
                worst = max(worst, abs(r0['rules'][ours]['all'][key] - block['evaluation'][theirs][key]))
    assert worst < 1e-12, worst
    return dict(max_abs_difference=worst)


def seed_check():
    seeds = [rb.seeds(r) for r in range(rb.REPEATS)]
    assert len(set(s for s, _ in seeds)) == rb.REPEATS and seeds[0] == (bw.SPLIT_SEED, None)
    noise_seeds = [n for _, n in seeds[1:]]
    assert len(set(noise_seeds)) == rb.REPEATS - 1 and min(np.diff(sorted(noise_seeds))) > 2 * 1000
    splits = [tuple(bw.split_samples(1000, seed=s)[0]) for s, _ in seeds]
    assert len(set(splits)) == rb.REPEATS
    for s, _ in seeds:
        cal, ev = bw.split_samples(1000, seed=s)
        assert len(cal) == len(ev) == 500 and not set(cal) & set(ev)
    return dict(repeats=rb.REPEATS, distinct=True)


def aligned_rule_check():
    probability = np.full((1, 10, 4), .95)
    probability[0, 9] = [.55, .65, .7, .7]
    success = np.ones((1, 10, 4), dtype=bool)
    success[0, 9, 0] = False
    plants = np.arange(10)
    assert bw.calibrate_threshold(probability, success, plants)[0] == .5
    threshold, outcome, reached = rb.aligned_threshold(probability, success, plants)
    assert threshold == .6 and reached and abs(outcome['mean_completion_min'] - 20.2) < 1e-12, (threshold, outcome)
    return dict(first_feasible=.5, aligned=.6)


def aligned_leakage_check(tables, saved):
    split = bw.split_samples(1000)
    observed = bw.noisy_observations(tables, br.NOISE_SD)
    flipped = dict(tables)
    flipped['rise'] = tables['rise'].copy()
    flipped['rise'][:, split[1], :] = np.where(flipped['rise'][:, split[1], :] > 0, -1., 1.)
    choices = rb.aligned_choices(flipped, br.NOISE_SD, split, observed)
    assert choices == saved['repeats']['0'][f'{br.NOISE_SD:g}']['aligned'], choices
    return dict(evaluation_labels_flipped=True, aligned_choices_unchanged=True)


def recompute_check(saved, r=7):
    fresh = rb.run_repeat(r, br.NOISE_SD)
    old = saved['repeats'][str(r)][f'{br.NOISE_SD:g}']
    worst = 0.
    for rule, block in old['rules'].items():
        for part, row in block.items():
            for key, value in row.items():
                worst = max(worst, abs(fresh['rules'][rule][part][key] - value))
    assert fresh['choices'] == old['choices'] and fresh['aligned'] == old['aligned'] and worst < 1e-12, worst
    return dict(repeat=r, max_abs_difference=worst)


def main():
    rb.load_protocol()
    tables = dict(np.load(br.TABLES))
    saved = json.loads(rb.RESULTS.read_text(encoding='utf-8'))
    report = dict(status='passed', protocol_sha256=rb.protocol_sha256(), version3=version3_check(saved),
                  seeds=seed_check(), aligned_rule=aligned_rule_check(),
                  aligned_leakage=aligned_leakage_check(tables, saved), recomputation=recompute_check(saved),
                  scope='Stability of one simulated model structure; not evidence about measured cells.')
    (br.OUT / 'waiting_robustness_verification_results.json').write_text(json.dumps(report, indent=2) + '\n', newline='\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    sys.exit(main())
