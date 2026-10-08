"""Audit of E3, from response prediction to waiting decisions (configs/b3_predict_decide_protocol.json).

Checks: freeze hash and pinned inputs; rho thresholded at kappa reproduces the success table; causality of
features and predictions; calibration maps depend on reference receivers only; history_mean equals a direct
average; the certified fixed comparator equals E1; leakage (inverted test outcomes leave every threshold
unchanged); split 0 recomputed from scratch; summary and statements re-derived with independent code.
Usage: python verify_b3_predict_decide.py
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')  # one BLAS thread per process: no oversubscription in pools
import hashlib
import json

import numpy as np
from scipy import stats

import b3_predict_decide as bp
import b3_readiness as br
import b3_waiting as bw
import b3_waiting_strict as st
from verify_b3_waiting_strict import close

OUT = br.OUT / 'predict_decide_verification_results.json'


def freeze_check(protocol):
    freeze = json.loads(bp.FREEZE.read_text(encoding='utf-8'))
    assert hashlib.sha256(bp.PROTOCOL.read_bytes()).hexdigest() == freeze['protocol_sha256']
    for path, digest in protocol['inputs_sha256'].items():
        assert hashlib.sha256((br.ROOT.parent / path).read_bytes()).hexdigest() == digest, path
    return dict(protocol_sha256=freeze['protocol_sha256'])


def same_or_nan(a, b):
    return np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(a[~np.isnan(a)], b[~np.isnan(b)], rtol=0, atol=1e-12)


def causality_check(tables, success):
    R, C, _ = st.roles(1)
    observed = bw.noisy_observations(tables, bp.NOISE, seed=st.noise_seed(1))
    rho = bp.rho_table(tables)
    bundle = bp.fit_bundle(tables, observed, rho, success, R)
    plants = C[:12]
    full = bp.predict(bundle, tables, observed, plants)
    for cut in (0, 20, 45):
        masked = observed.copy()
        masked[:, :, cut + 1:] = np.nan
        part = bp.predict(bundle, tables, masked, plants)
        for name in bp.PREDICTORS:
            assert same_or_nan(full[name][0][..., :cut + 1], part[name][0][..., :cut + 1]), (name, cut, 'rho')
            upto = min(cut + 1, full[name][1].shape[-1])
            assert same_or_nan(full[name][1][..., :upto], part[name][1][..., :upto]), (name, cut, 'forecast')
        f_full, f_part = bp.features(observed[1, plants], 1), bp.features(masked[1, plants], 1)
        assert same_or_nan(f_full[:, :cut + 1], f_part[:, :cut + 1]), cut
    return dict(cuts=[0, 20, 45], predictors=list(bp.PREDICTORS))


def reference_only_check(tables, success):
    R, C, T = st.roles(2)
    observed = bw.noisy_observations(tables, bp.NOISE, seed=st.noise_seed(2))
    rho = bp.rho_table(tables)
    masked_rho, masked_success = rho.copy(), success.astype(float)
    for role in (C, T):
        masked_rho[:, role, :] = np.nan
        masked_success[:, role, :] = np.nan
    a = bp.fit_bundle(tables, observed, rho, success, R)
    b = bp.fit_bundle(tables, observed, masked_rho, masked_success, R)
    for name in bp.PREDICTORS:
        for h in range(2):
            assert np.array_equal(a['maps'][name][h].x, b['maps'][name][h].x), name
            assert np.array_equal(a['maps'][name][h].y, b['maps'][name][h].y), name
    assert a['lambda'] == b['lambda']
    mean = bp.predict(a, tables, observed, T[:5])['history_mean'][0]
    direct = rho[:, R, :].mean(1)
    assert np.allclose(mean, np.broadcast_to(direct[:, None, :], mean.shape), atol=1e-12)
    return dict(maps_use_reference_receivers_only=True, history_mean_is_direct_average=True)


def rho_check(tables, success):
    assert np.array_equal(bp.rho_table(tables) >= bp.KAPPA, success)
    return dict(rho_reproduces_success=True)


def comparator_check(result):
    e1 = json.loads(st.RESULTS.read_text(encoding='utf-8'))['repeats']
    for r in range(st.SPLITS):
        mine = result['repeats'][str(r)]['fixed']['choice']
        theirs = e1[str(r)]['0.005']['arms']['certified']['choices']['fixed']
        assert {h: mine[h]['wait_min'] for h in br.HISTORIES} == {h: theirs[h]['wait_min'] for h in br.HISTORIES}, r
    return dict(fixed_choices_equal_e1=True)


def leakage_check(tables, success, saved):
    _, _, T = st.roles(0)
    flipped = success.copy()
    flipped[:, T, :] = ~flipped[:, T, :]
    rerun = bp.run_split(0, tables=tables, success=flipped)
    close({n: rerun['predictors'][n]['choice'] for n in bp.PREDICTORS},
          {n: saved['predictors'][n]['choice'] for n in bp.PREDICTORS})
    return dict(test_outcomes_inverted_thresholds_unchanged=True)


def summary_check(result):
    repeats, summary = result['repeats'], result['summary']
    rows = [repeats[str(r)] for r in range(st.SPLITS)]
    for name in bp.PREDICTORS:
        for metric in ('forecast_rmse', 'rho_rmse_all', 'rho_rmse_late', 'brier', 'ece'):
            assert abs(summary['metrics'][name][metric]['median']
                       - float(np.median([row['predictors'][name]['metrics'][metric] for row in rows]))) < 1e-12
        meets = sum(all(row['predictors'][name]['rules'][h]['success_rate'] >= .9 for h in br.HISTORIES) for row in rows)
        assert summary['meets_per_history'][name] == meets
    rho_a, rho_b = [], []
    for row in rows:
        completion = [row['predictors'][n]['rules']['all']['mean_completion_min'] for n in bp.PREDICTORS]
        rho_a.append(stats.spearmanr([row['predictors'][n]['metrics']['forecast_rmse'] for n in bp.PREDICTORS],
                                     completion).statistic)
        rho_b.append(stats.spearmanr([row['predictors'][n]['metrics']['rho_rmse_all'] for n in bp.PREDICTORS],
                                     completion).statistic)
    assert abs(result['statements']['E3-S2']['forecast_vs_completion_median'] - float(np.median(rho_a))) < 1e-12
    assert abs(result['statements']['E3-S2']['window_vs_completion_median'] - float(np.median(rho_b))) < 1e-12
    order = sorted(bp.PREDICTORS, key=lambda n: repeats['0']['predictors'][n]['metrics']['forecast_rmse'])
    assert result['statements']['E3-S1']['by_forecast_rmse'] == order
    return dict(summary_rederived=True, statements_rederived=True)


def main():
    protocol = bp.load_protocol()
    result = json.loads(bp.RESULTS.read_text(encoding='utf-8'))
    assert result['status'] == 'complete' and result['protocol_sha256'] == hashlib.sha256(bp.PROTOCOL.read_bytes()).hexdigest()
    tables = dict(np.load(br.TABLES))
    success = br.success_table(tables)
    checks = dict(freeze=freeze_check(protocol), rho=rho_check(tables, success),
                  causality=causality_check(tables, success), reference_only=reference_only_check(tables, success),
                  comparator=comparator_check(result), leakage=leakage_check(tables, success, result['repeats']['0']),
                  summary=summary_check(result))
    close(bp.run_split(0, tables=tables), result['repeats']['0'], 'split 0')
    checks['recompute'] = dict(recomputed_split=0)
    OUT.write_text(json.dumps(dict(status='passed', checks=checks), indent=2) + '\n', newline='\n')
    print('E3 prediction-to-decision verification: passed')


if __name__ == '__main__':
    main()
