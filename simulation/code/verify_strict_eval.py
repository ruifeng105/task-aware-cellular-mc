"""Unit tests of strict_eval (shared helpers of the strict waiting evaluations E1-E4).
Usage: python verify_strict_eval.py
"""
import numpy as np
from scipy import stats

import strict_eval as se


def test_cp_lower_matches_beta_quantile():
    for k, n in ((0, 10), (9, 10), (300, 316), (316, 316), (1, 1)):
        reference = 0. if k == 0 else stats.beta.ppf(.025, k, n - k + 1)
        assert abs(se.cp_lower(k, n, .025) - reference) < 1e-12, (k, n)


def test_cp_lower_is_dual_to_binomial_test():
    for n in (50, 100, 316):
        for k in range(int(.8 * n), n + 1):
            assert (se.cp_lower(k, n, .025) >= .9) == (se.binom_pvalue(k, n, .9) <= .025), (k, n)


def test_fixed_sequence_stops_at_first_non_rejection():
    assert se.fixed_sequence([(316, 316), (310, 316), (250, 316), (316, 316)], .9, .025) == 2
    assert se.fixed_sequence([(250, 316), (316, 316)], .9, .025) == 0
    assert se.fixed_sequence([], .9, .025) == 0


def test_distinct_samples_keeps_first_occurrence():
    posterior = np.array([[1, 2], [1, 2], [3, 4], [5, 6], [7, 8], [9, 9], [1, 2]], float)
    assert se.distinct_samples(posterior).tolist() == [0, 2, 3, 4, 5]


def test_three_way_split_is_disjoint_and_reproducible():
    ids = np.arange(5, 953)
    r, c, t = se.three_way_split(ids, seed=1)
    assert len(r) == len(c) == len(t) == 316
    assert not (set(r) & set(c) or set(r) & set(t) or set(c) & set(t))
    assert set(r) | set(c) | set(t) == set(ids.tolist())
    assert all(np.array_equal(a, b) for a, b in zip((r, c, t), se.three_way_split(ids, seed=1)))
    assert not np.array_equal(r, se.three_way_split(ids, seed=2)[0])


def _option(label, successes, n, completion):
    ok = np.zeros((2, n), bool)
    ok[:, :successes] = True
    return (label, ok, np.where(ok, completion, 140.))


def test_certified_choice_falls_back_when_nothing_is_admitted():
    chain = [_option(dict(threshold=.99), 40, 50, 100.)]
    choice = se.certified_choice([chain], 0, .9, .025)
    assert choice['certified'] is False and choice['fallback'] == 'probe_at_120'
    assert 'threshold' not in choice


def test_certified_choice_takes_fastest_admitted_parameter():
    chain = [_option(dict(threshold=.99), 316, 316, 110.), _option(dict(threshold=.98), 312, 316, 100.),
             _option(dict(threshold=.97), 250, 316, 60.), _option(dict(threshold=.96), 316, 316, 50.)]
    choice = se.certified_choice([chain], 1, .9, .025)
    assert choice['certified'] is True and choice['threshold'] == .98, choice
    assert choice['calibration']['admitted'] == 2


def test_certified_choice_splits_alpha_over_chains():
    n = 316
    k = next(k for k in range(n + 1) if se.binom_pvalue(k, n, .9) <= .025)   # passes at .025, not at .0125
    assert se.binom_pvalue(k, n, .9) > .0125
    one = [_option(dict(tau_min=2., threshold=.9), k, n, 90.)]
    other = [_option(dict(tau_min=4., threshold=.9), n, n, 95.)]
    assert se.certified_choice([one], 0, .9, .025)['tau_min'] == 2.
    assert se.certified_choice([one, other], 0, .9, .025)['tau_min'] == 4.


def test_paired_bootstrap_interval_contains_mean():
    diff = np.random.default_rng(3).normal(-5., 2., 300)
    low, high = se.paired_bootstrap(diff, 2000, 11)
    assert low < diff.mean() < high and high - low < 1.
    assert se.paired_bootstrap(diff, 2000, 11) == [low, high]


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for test in tests:
        test()
    print(f'strict_eval helpers: passed ({len(tests)} tests)')


if __name__ == '__main__':
    main()
