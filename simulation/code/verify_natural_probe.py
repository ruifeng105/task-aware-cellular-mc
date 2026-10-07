"""Audit the natural-probe check on measured cells.

Unit-tests the rise operator, the AUC and the estimators on synthetic data,
checks that the decision frame precedes each probe onset, and recomputes the
whole analysis (B3 references are re-simulated) for comparison with the
saved results.
"""
import json
import sys

import numpy as np

import natural_probe as npb

VERIFY = npb.OUT / 'natural_probe_verification_results.json'


def operator_check():
    t = np.arange(1., 60., 2.)
    y = np.where(t > 23, 1.2, 1.0)[None]
    assert np.isclose(npb.rise(t, y, 23., 1.)[0], .2)                  # step after onset: full rise
    assert np.isclose(npb.rise(t, y * 0 + 1.05, 0., 1.)[0], .05)        # no frame before onset: default level
    ramp = (1 + .01 * t)[None]
    assert np.isclose(npb.rise(t, ramp, 23., 1.)[0], .01 * (43 - 22))   # pre level = mean of frames 21 and 23
    assert npb.last_index(t, 46.) == int(np.flatnonzero(t == 45.)[0])
    return dict(rise_operator=True, decision_frame_precedes_onset=True)


def auc_check():
    rng = np.random.default_rng(3)
    score, outcome = rng.random(200).round(1), rng.random(200) < .4
    pos, neg = score[outcome], score[~outcome]
    brute = np.mean([(p > n) + .5 * (p == n) for p in pos for n in neg])
    assert np.isclose(npb.auc(score, outcome), brute) and npb.auc(score, np.ones(200, bool)) is None
    return dict(auc_matches_pairwise_count=True)


def estimator_check():
    rng = np.random.default_rng(4)
    t = np.arange(0., 40., 2.)
    simulated = 1 + .05 * rng.random((30, t.size))
    success = rng.random(30) < .5
    measured = simulated[:3] + .001
    predictions, ess = npb.estimates(t, measured, simulated, success, 5, .005)
    assert np.allclose(predictions['constant'], success.mean())
    assert ((predictions['history'] >= 0) & (predictions['history'] <= 1)).all() and (ess >= 1).all()
    later = measured.copy()
    later[:, 6:] += .3
    again, _ = npb.estimates(t, later, simulated, success, 5, .005)
    assert all(np.allclose(predictions[n], again[n]) for n in predictions)   # frames after the decision frame unused
    return dict(constant_is_b3_readiness=True, later_frames_unused=True)


def recompute_check(saved):
    fresh = json.loads(json.dumps(npb.compute()))
    worst = 0.

    def walk(a, b, path=''):
        nonlocal worst
        if isinstance(a, dict):
            assert a.keys() == b.keys(), path
            for k in a:
                walk(a[k], b[k], f'{path}/{k}')
        elif isinstance(a, list):
            assert len(a) == len(b), path
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f'{path}/{i}')
        elif isinstance(a, (int, float)) and not isinstance(a, bool) and a is not None:
            worst = max(worst, abs(a - b))
        else:
            assert a == b, path
    walk({k: v for k, v in fresh.items() if k not in ('frozen_utc',)}, {k: v for k, v in saved.items() if k not in ('frozen_utc',)})
    assert worst < 1e-9, worst
    return dict(max_abs_difference=worst)


def main():
    npb.load_protocol()
    saved = json.loads(npb.RESULTS.read_text(encoding='utf-8'))
    checks = dict(operator=operator_check(), auc=auc_check(), estimators=estimator_check(), recompute=recompute_check(saved))
    result = dict(status='passed', protocol_sha256=npb.protocol_sha256(), checks=checks)
    VERIFY.write_text(json.dumps(result, indent=2) + '\n', newline='\n')
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
