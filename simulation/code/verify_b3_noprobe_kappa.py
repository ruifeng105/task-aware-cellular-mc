"""Audit the kappa-dependent no-probe control and causal rescoring.

Checks that kappa 0.5 reproduces the frozen no-probe study, that the
recomputed stop times reproduce the frozen sensitivity and conditioned-wait
results at every kappa, that labels are nested in kappa, and recomputes the
kappa-0.4 no-probe labels from the raw no-probe trajectories.
"""
import json
import sys

import numpy as np

import b3_noprobe as npb
import b3_noprobe_kappa as nk
import b3_readiness as br
import b3_waiting_baselines as bl
import b3_waiting_sensitivity as sv


def frozen_check(saved):
    frozen = json.loads(npb.RESULTS.read_text(encoding='utf-8'))
    states = saved['kappa']['0.5']['states']
    assert {k: states['agreement'][k] for k in frozen['agreement']} == frozen['agreement']
    assert states['success_without_probe']['fraction'] == frozen['success_without_probe']['fraction']
    assert abs(saved['max_rise_without_probe_over_naive'] - frozen['success_without_probe']['max_rise_without_probe_over_naive']) < 1e-12
    worst = max(abs(saved['kappa']['0.5']['rescored']['0.005'][label][name][key] - frozen['rescored'][label][name][key])
                for label in ('original', 'causal') for name in frozen['rescored'][label] for key in ('success_rate', 'mean_completion_min'))
    assert worst < 1e-12, worst
    return dict(agreement_equal=True, rescoring_max_abs_difference=worst)


def stop_time_check(saved):
    sensitivity = json.loads(sv.RESULTS.read_text(encoding='utf-8'))['cells']
    conditioned = json.loads(bl.RESULTS.read_text(encoding='utf-8'))['kappa']
    worst = 0.
    for kappa, block in saved['kappa'].items():
        for noise, rescored in block['rescored'].items():
            for name in ('current', 'smoothed', 'history', 'fixed'):
                frozen = sensitivity[kappa][noise]['rules'][name]['all']
                worst = max(worst, *(abs(rescored['original'][name][k] - frozen[k]) for k in ('success_rate', 'mean_completion_min')))
            frozen = conditioned[kappa]['evaluation']['all']
            worst = max(worst, *(abs(rescored['original']['conditioned'][k] - frozen[k]) for k in ('success_rate', 'mean_completion_min')))
            assert rescored['original']['oracle']['success_rate'] == sensitivity[kappa][noise]['rules']['oracle']['all']['success_rate']
    assert worst < 1e-12, worst
    return dict(max_abs_difference=worst)


def nesting_check(tables, arrays):
    previous = None
    for kappa in sv.KAPPAS:
        _, without, causal = nk.labels(tables, arrays, kappa)
        if previous is not None:
            assert not (without & ~previous[0]).any() and not (causal & ~previous[1]).any(), kappa
        previous = (without, causal)
    return dict(labels_nested_in_kappa=True)


def raw_check(tables, arrays, kappa=.4):
    noprobe, naive = arrays['noprobe'], tables['naive_rise'].astype(float)
    width = len(npb.WINDOW)
    rise = np.stack([noprobe[:, :, s:s + width].max(-1) - noprobe[:, :, s] for s in br.GRID.astype(int)], axis=2)
    direct = rise >= kappa * naive[None, :, None]
    _, without, _ = nk.labels(tables, arrays, kappa)
    assert np.array_equal(direct, without)
    return dict(kappa=kappa, states_without_probe=int(direct.sum()))


def main():
    nk.load_protocol()
    tables, arrays = dict(np.load(br.TABLES)), dict(np.load(npb.ARRAYS))
    saved = json.loads(nk.RESULTS.read_text(encoding='utf-8'))
    report = dict(status='passed', protocol_sha256=nk.protocol_sha256(), frozen=frozen_check(saved),
                  stop_times=stop_time_check(saved), nesting=nesting_check(tables, arrays), raw=raw_check(tables, arrays),
                  scope='Simulation of the published B3 model; not evidence about measured cells.')
    (br.OUT / 'noprobe_kappa_verification_results.json').write_text(json.dumps(report, indent=2) + '\n', newline='\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    sys.exit(main())
