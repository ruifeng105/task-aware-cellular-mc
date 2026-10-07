"""Audit the model-mismatch study of the calibrated waiting rules.

Implements the checks of configs/b3_waiting_mismatch_protocol.json: baseline
equal to the frozen frontier study; hidden-lag invariants (short history
unchanged, long-history success equal to the share of plants whose baseline
slack covers the lag, oracle shifted by the lag); slow-tail outcomes equal to
the baseline restricted to the subset; threshold-only recalibration at lag 0
equal to the baseline; leakage; one split recomputed; descriptive results
recomputed; summary and statements M1-M4 re-derived from the stored splits.
"""
import json
import sys

import numpy as np

import b3_readiness as br
import b3_waiting as bw
import b3_waiting_frontier as wf
import b3_waiting_mismatch as wm
import b3_waiting_robustness as rb

TOL = 1e-12


def close(a, b):
    """Recursive comparison of JSON-like values with numbers within TOL."""
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return a is not None and b is not None and abs(a - b) <= TOL
    return a == b


def baseline_check(repeats):
    frontier = json.loads(wf.RESULTS.read_text(encoding='utf-8'))['repeats']
    compared = 0
    for r in range(rb.REPEATS):
        for noise in (f'{n:g}' for n in wm.NOISES):
            for target, cell in repeats[str(r)][noise]['baseline'].items():
                frozen = frontier[str(r)][noise]['cells'][target]
                assert cell['choices'] == frozen['choices'], (r, noise, target)
                for key in ('rules', 'paired', 'gap_closure', 'meets_per_history'):
                    assert close(cell[key], frozen[key]), (r, noise, target, key)
                compared += 1
    return dict(cells_compared=compared)


def hidden_lag_check(repeats, success):
    checked = 0
    for r in range(rb.REPEATS):
        _, ev = bw.split_samples(success.shape[1], seed=rb.seeds(r)[0])
        onset = wm.onsets(success)[wm.LONG, ev]
        for noise in (f'{n:g}' for n in wm.NOISES):
            block = repeats[str(r)][noise]
            for target in block['baseline']:
                previous = {name: 1. for name in wf.RULES}
                for lag in wm.MC_LAGS:
                    rules = block['hidden_lag']['model_calibrated'][f'{lag}'][target]['rules']
                    for name in wf.RULES:
                        assert close(rules[name]['short'], block['baseline'][target]['rules'][name]['short']), (r, lag, name)
                        share = block['slack'][target][name]['long']['share_at_least'][f'{lag}']
                        assert abs(rules[name]['long']['success_rate'] - share) <= TOL, (r, noise, target, lag, name)
                        assert rules[name]['long']['success_rate'] <= previous[name] + TOL
                        previous[name] = rules[name]['long']['success_rate']
                    oracle = float((onset <= br.GRID[-1] - lag).mean())
                    assert abs(rules['oracle']['long']['success_rate'] - oracle) <= TOL, (r, lag)
                    checked += 1
    return dict(cells_checked=checked)


def split_parts(r, noise):
    tables = dict(np.load(br.TABLES))
    success = br.success_table(tables)
    split_seed, noise_seed = rb.seeds(r)
    cal, ev = bw.split_samples(success.shape[1], seed=split_seed)
    observed = bw.noisy_observations(tables, noise, seed=noise_seed)
    on_cal = bw.estimator_probabilities(tables, success, noise, cal, cal, observed)
    on_ev = bw.estimator_probabilities(tables, success, noise, cal, ev, observed)
    return tables, success, cal, ev, on_cal, on_ev


def slow_and_recalibration_check(repeats, r=3, noise=.005):
    tables, success, cal, ev, on_cal, on_ev = split_parts(r, noise)
    deadline = br.DEADLINE_MIN
    stored = repeats[str(r)][f'{noise:g}']
    base_cal = wf.candidates(on_cal, success, cal, deadline)
    base_ev = wf.candidates(on_ev, success, ev, deadline)
    for target, cell in stored['baseline'].items():
        choices = cell['choices']
        outcomes = wf.evaluated(base_ev, choices)
        for cut in wm.ONSET_CUTS:
            keep = wm.slow_mask(success, cut)[ev]
            restricted = {name: rb.summarise(ok[:, keep], completion[:, keep]) for name, (ok, completion) in outcomes.items()}
            slow = stored['slow_posterior_tail'][f'{cut:g}']['model_calibrated'][target]['rules']
            assert all(close(restricted[name], slow[name]) for name in wf.RULES), (cut, target)
    oracle = wf.oracle_outcomes(success, ev, deadline)
    plain, _ = wm.recalibrated_cells(on_cal, on_ev, success, cal, ev, deadline, oracle)
    for target, cell in plain.items():
        assert cell['choices'] == stored['baseline'][target]['choices'], target
        assert close(cell['rules'], stored['baseline'][target]['rules']), target
    return dict(r=r, noise=noise, slow_restriction=True, threshold_only_lag0_equals_baseline=True)


def leakage_check(repeats, r=5, noise=.005, lag=8):
    tables, success, cal, ev, on_cal, on_ev = split_parts(r, noise)
    plant = wm.lagged(success, lag)
    flipped = plant.copy()
    flipped[:, ev, :] = ~flipped[:, ev, :]
    deadline = br.DEADLINE_MIN
    oracle = wf.oracle_outcomes(plant, ev, deadline)
    stored = repeats[str(r)][f'{noise:g}']['hidden_lag']
    plain_cal = wf.candidates(on_cal, flipped, cal, deadline)
    delay_cal = wm.delay_candidates(on_cal, flipped, cal, deadline)
    for target in map(lambda t: f'{t:.2f}', wm.TARGETS):
        for arm, options, chooser in (('outcome_recalibrated', plain_cal, wf.choose),
                                      ('outcome_recalibrated_delay', delay_cal, wm.choose_delay)):
            choices = {name: {hist: chooser(options[name], h, float(target)) for h, hist in enumerate(br.HISTORIES)}
                       for name in wf.RULES}
            assert json.loads(json.dumps(choices)) == stored[arm][f'{lag}'][target]['choices'], (arm, target)
    return dict(r=r, noise=noise, lag=lag, evaluation_outcomes_inverted=True, choices_unchanged=True)


def recompute_check(repeats, r=7, noise=.01):
    fresh = wm.run_repeat(r, noise)
    assert fresh == repeats[str(r)][f'{noise:g}']
    return dict(r=r, noise=noise, identical=True)


def descriptive_check(saved, tables, success):
    fresh = json.loads(json.dumps(wm.descriptive(tables, success)))
    assert close(fresh, saved['descriptive'])
    return dict(identical=True, shift_mixed_min=fresh['shift_equivalents']['mixed']['shift_equivalent_min'],
                null_median=fresh['no_response_null']['median'])


def statements_check(saved, repeats):
    """Summary re-derived from the stored splits; statements M1-M4 recomputed from the per-split success rates and
    paired differences with code independent of b3_waiting_mismatch.statements (added after the run)."""
    assert close(json.loads(json.dumps(wm.summaries(repeats))), saved['summary'])
    st, t, p = saved['statements'], wm.PRIMARY_TARGET, wm.PRIMARY_NOISE
    splits = [str(r) for r in range(rb.REPEATS)]

    def cells(noise, *path):
        out = []
        for r in splits:
            node = repeats[r][noise]
            for key in path:
                node = node[key]
            out.append(node[t])
        return out

    def meets(cell, name):
        return all(cell['rules'][name][h]['success_rate'] >= br.Q for h in br.HISTORIES)

    lags = list(wm.MC_LAGS)
    for name in wf.RULES:
        ok = [sum(meets(c, name) for c in cells(p, 'hidden_lag', 'model_calibrated', f'{lag}')) >= wm.ROBUST_SPLITS
              for lag in lags]
        expected = None if not ok[0] else lags[-1] if all(ok) else lags[ok.index(False) - 1]
        assert st['M1']['tolerated_lag_min'][name] == expected, ('M1', name)
        for lag in lags:
            values = [c['rules'][name]['long']['success_rate'] for c in cells(p, 'hidden_lag', 'model_calibrated', f'{lag}')]
            assert abs(float(np.median(values)) - st['M1']['success_after_long_median'][f'{lag}'][name]) <= TOL, ('M1', lag)
    for noise in st['M2']:
        base = cells(noise, 'baseline')
        for lag, row in st['M2'][noise].items():
            seen = cells(noise, 'visible_lag', 'model_calibrated', lag)
            d = [(v['rules']['history']['long']['success_rate'] - b['rules']['history']['long']['success_rate'])
                 - (v['rules']['smoothed']['long']['success_rate'] - b['rules']['smoothed']['long']['success_rate'])
                 for b, v in zip(base, seen)]
            positive, negative = sum(x > 0 for x in d), sum(x < 0 for x in d)
            verdict = ('B3 belief more robust' if positive >= wm.ROBUST_SPLITS else
                       'smoothing more robust' if negative >= wm.ROBUST_SPLITS else 'no consistent difference')
            ess = [repeats[r][noise]['ess']['visible_lag'][lag]['90']['median'] for r in splits]
            assert (row['positive'], row['negative'], row['verdict']) == (positive, negative, verdict), ('M2', noise, lag)
            assert abs(row['D']['median'] - float(np.median(d))) <= TOL and abs(row['ess_90']['median'] - float(np.median(ess))) <= TOL
    for scenario, arms in st['M3'].items():
        for arm, by_lag in arms.items():
            for lag, row in by_lag.items():
                group = cells(p, scenario, arm, lag)
                assert row['meets_per_history'] == {name: sum(meets(c, name) for c in group) for name in wf.RULES}, ('M3', lag)
                assert row['feasible_calibration'] == sum(c['feasible_calibration'] for c in group)
                for name in ('history', 'smoothed'):
                    diffs = [c['paired'][f'{name}-fixed/all']['completion'] for c in group
                             if c['target_reached'][name] and c['target_reached']['fixed']]
                    entry = row[f'{name}-fixed']
                    assert (entry['splits'], entry['faster']) == (len(diffs), sum(x < 0 for x in diffs)), ('M3', name, lag)
                    assert (entry['median'] is None) if not diffs else abs(entry['median'] - float(np.median(diffs))) <= TOL
    for cut, arms in st['M4'].items():
        for arm, row in arms.items():
            group = cells(p, 'slow_posterior_tail', cut, arm)
            assert row['meets_per_history'] == {name: sum(meets(c, name) for c in group) for name in wf.RULES}, ('M4', cut, arm)
            for name in ('history', 'smoothed', 'current'):
                for h in br.HISTORIES:
                    diffs = [c['rules'][name][h]['success_rate'] - c['rules']['fixed'][h]['success_rate'] for c in group]
                    entry = row[f'{name}-fixed/{h}']
                    assert entry['positive'] == sum(x > 0 for x in diffs), ('M4', cut, arm, name, h)
                    assert abs(entry['success_difference']['median'] - float(np.median(diffs))) <= TOL
    return dict(summary_rederived=True, statements_recomputed=['M1', 'M2', 'M3', 'M4'])


def main():
    wm.load_protocol()
    tables = dict(np.load(br.TABLES))
    success = br.success_table(tables)
    saved = json.loads(wm.RESULTS.read_text(encoding='utf-8'))
    splits_raw = wm.SPLITS.read_bytes()
    assert wm.hashlib.sha256(splits_raw).hexdigest() == saved['splits_sha256']
    repeats = json.loads(splits_raw)['repeats']
    report = dict(status='passed', protocol_sha256=wm.protocol_sha256(), baseline=baseline_check(repeats),
                  hidden_lag=hidden_lag_check(repeats, success), slow_and_recalibration=slow_and_recalibration_check(repeats),
                  leakage=leakage_check(repeats), recomputation=recompute_check(repeats),
                  descriptive=descriptive_check(saved, tables, success), statements=statements_check(saved, repeats),
                  scope='Simulation of the published B3 model with stylized recovery errors; not evidence about measured cells.')
    (br.OUT / 'waiting_mismatch_verification_results.json').write_text(json.dumps(report, indent=2) + '\n', newline='\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    sys.exit(main())
