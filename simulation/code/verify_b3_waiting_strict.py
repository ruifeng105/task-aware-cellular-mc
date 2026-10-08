"""Audit of E1, the strict reliability evaluation (configs/b3_waiting_strict_protocol.json).

Checks: freeze hash and pinned inputs; disjoint, distinct receiver roles; the candidate tables equal those of
b3_waiting_frontier; leakage (inverted test outcomes leave every choice unchanged); fallback (a calibration set
without successes makes every certified choice a 120-min probe, scored as such); splits 0 and 7 recomputed from
scratch; the 20-split summary and statements E1-S1 to E1-S4 re-derived with independent code.
Usage: python verify_b3_waiting_strict.py
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')  # one BLAS thread per process: no oversubscription in pools
import hashlib
import json

import numpy as np

import b3_model as b3
import b3_readiness as br
import b3_waiting as bw
import b3_waiting_frontier as fr
import b3_waiting_strict as st
import strict_eval as se

OUT = br.OUT / 'waiting_strict_verification_results.json'


def close(a, b, path=''):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), (path, sorted(a), sorted(b))
        for k in a:
            close(a[k], b[k], f'{path}/{k}')
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            close(x, y, f'{path}/{i}')
    elif isinstance(a, float) or isinstance(b, float):
        assert (a is None and b is None) or abs(a - b) <= 1e-12 + 1e-12 * abs(b), (path, a, b)
    else:
        assert a == b, (path, a, b)


def freeze_check(protocol):
    freeze = json.loads(st.FREEZE.read_text(encoding='utf-8'))
    assert hashlib.sha256(st.PROTOCOL.read_bytes()).hexdigest() == freeze['protocol_sha256']
    for path, digest in protocol['inputs_sha256'].items():
        assert hashlib.sha256((br.ROOT.parent / path).read_bytes()).hexdigest() == digest, path
    return dict(protocol_sha256=freeze['protocol_sha256'], inputs_pinned=len(protocol['inputs_sha256']))


def role_check():
    posterior = b3.load_posterior()
    for r in range(st.SPLITS):
        R, C, T = st.roles(r)
        assert len(R) == len(C) == len(T) == 316
        union = np.concatenate([R, C, T])
        assert len(set(union.tolist())) == 948
        assert np.unique(posterior[union], axis=0).shape[0] == 948, r   # no twin anywhere in the population
    return dict(splits=st.SPLITS, receivers_per_role=316, distinct=948)


def candidate_check(tables, success):
    R, C, _ = st.roles(3)
    observed = bw.noisy_observations(tables, .005, seed=st.noise_seed(3))
    probability = bw.estimator_probabilities(tables, success, .005, R, C, observed)
    mine, waits = st.options_and_waits(probability, success, C, br.DEADLINE_MIN)
    theirs = fr.candidates(probability, success, C, br.DEADLINE_MIN)
    for name in theirs:
        assert [o[0] for o in mine[name]] == [o[0] for o in theirs[name]], name
        for a, b in zip(mine[name], theirs[name]):
            assert np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2]), name
    ok, completion = mine['fixed'][-1][1], mine['fixed'][-1][2]
    assert np.array_equal(waits[('fixed', (120.,))], np.full(ok.shape, 120.))
    return dict(candidates_equal_frontier=True)


def strip(arms):
    return {arm: block['choices'] for arm, block in arms.items()}


def leakage_check(tables, success, saved):
    _, _, T = st.roles(0)
    flipped = success.copy()
    flipped[:, T, :] = ~flipped[:, T, :]
    rerun = st.run_split(0, .005, tables=tables, success=flipped)
    close(strip(rerun['arms']), strip(saved['arms']))
    return dict(test_outcomes_inverted_choices_unchanged=True)


def fallback_check(tables, success):
    _, C, T = st.roles(0)
    none = success.copy()
    none[:, C, :] = False
    rerun = st.run_split(0, .005, tables=tables, success=none)
    choices = rerun['arms']['certified']['choices']
    assert all(choices[name][hist]['certified'] is False and choices[name][hist]['fallback'] == se.FALLBACK
               for name in st.RULES for hist in br.HISTORIES)
    k120 = int(np.flatnonzero(br.GRID == 120.)[0])
    for name in st.RULES:
        for h, hist in enumerate(br.HISTORIES):
            block = rerun['arms']['certified']['rules'][name][hist]
            assert block['successes'] == int(none[h, T, k120].sum()) and block['mean_wait_min'] == 120., (name, hist)
    return dict(all_certified_choices_fall_back=True, scored_at_120=True)


def recompute_check(tables, saved):
    for r in (0, 7):
        for noise in st.NOISES:
            close(st.run_split(r, noise, tables=tables), saved[str(r)][f'{noise:g}'], f'split {r} noise {noise}')
    return dict(recomputed_splits=[0, 7])


def summary_check(result):
    repeats = result['repeats']
    for noise in (f'{n:g}' for n in st.NOISES):
        for arm in st.ARMS:
            rows = [repeats[str(r)][noise]['arms'][arm] for r in range(st.SPLITS)]
            block = result['summary'][noise][arm]
            for name in st.RULES:
                meets = sum(all(row['rules'][name][h]['success_rate'] >= .9 for h in br.HISTORIES) for row in rows)
                joint = sum(all(row['rules'][name][h]['lower_975'] >= .9 for h in br.HISTORIES) for row in rows)
                certified = sum(all(row['choices'][name][h]['certified'] for h in br.HISTORIES) for row in rows) \
                    if arm == 'certified' else None
                assert block['meets_per_history'][name] == meets, (noise, arm, name)
                assert block['joint_bound'][name] == joint, (noise, arm, name)
                assert block['certified_both'][name] == certified, (noise, arm, name)
                values = [row['rules'][name]['all']['mean_completion_min'] for row in rows]
                assert abs(block['completion'][name]['median'] - float(np.median(values))) < 1e-12
            for name in st.ADAPTIVE:
                diffs = [row['paired'][name]['all']['completion'] for row in rows]
                assert block['faster'][name] == sum(d < 0 for d in diffs)
                assert abs(block['paired_completion'][name]['median'] - float(np.median(diffs))) < 1e-12
    s = result['statements']
    primary = repeats['0']['0.005']['arms']
    for name in st.RULES:
        for hist in br.HISTORIES:
            row = primary['certified']['rules'][name][hist]
            assert s['E1-S1'][name][hist]['success_rate'] == row['success_rate']
            assert abs(s['E1-S1'][name][hist]['lower_95'] - se.cp_lower(row['successes'], row['n'], .05)) < 1e-12
        assert s['E1-S1'][name]['splits_meeting'] == result['summary']['0.005']['certified']['meets_per_history'][name]
        assert s['E1-S3'][name] == result['summary']['0.005']['plugin_090']['meets_per_history'][name]
    for name in st.ADAPTIVE:
        assert s['E1-S2'][name]['split0'] == primary['certified']['paired'][name]['all']['completion']
        cost = [repeats[str(r)]['0.005']['arms']['certified']['rules'][name]['all']['mean_completion_min']
                - repeats[str(r)]['0.005']['arms']['plugin_094']['rules'][name]['all']['mean_completion_min']
                for r in range(st.SPLITS)]
        assert abs(s['E1-S4'][name] - float(np.median(cost))) < 1e-12
    return dict(summary_rederived=True, statements_rederived=True)


def main():
    protocol = st.load_protocol()
    result = json.loads(st.RESULTS.read_text(encoding='utf-8'))
    assert result['status'] == 'complete' and result['protocol_sha256'] == hashlib.sha256(st.PROTOCOL.read_bytes()).hexdigest()
    tables = dict(np.load(br.TABLES))
    success = br.success_table(tables)
    checks = dict(freeze=freeze_check(protocol), roles=role_check(), candidates=candidate_check(tables, success),
                  leakage=leakage_check(tables, success, result['repeats']['0']['0.005']),
                  fallback=fallback_check(tables, success), recompute=recompute_check(tables, result['repeats']),
                  summary=summary_check(result))
    OUT.write_text(json.dumps(dict(status='passed', checks=checks), indent=2) + '\n', newline='\n')
    print('E1 strict evaluation verification: passed')


if __name__ == '__main__':
    main()
