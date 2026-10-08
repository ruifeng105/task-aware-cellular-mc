"""Audit of E4, task-definition and mismatch-form sensitivity (configs/b3_waiting_scope_protocol.json).

Checks: freeze hash and pinned inputs; the 30-min probe-window tables reproduce the readiness tables at W = 20
and the rise grows with W; the baseline cell equals the certified arm of E1 in every split; every mismatch form
reduces to the model at its neutral level and is monotone in its severity; the anchors reproduce the
shift-equivalent crossing of the mismatch study; leakage (inverted test outcomes leave every re-certified choice
unchanged); three jobs recomputed from scratch; summary and statements re-derived with independent code.
Usage: python verify_b3_waiting_scope.py
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')  # one BLAS thread per process: no oversubscription in pools
import hashlib
import json

import numpy as np

import b3_readiness as br
import b3_waiting_mismatch as wm
import b3_waiting_scope as sc
import b3_waiting_strict as st
from verify_b3_waiting_strict import close

OUT = br.OUT / 'waiting_scope_verification_results.json'


def freeze_check(protocol):
    freeze = json.loads(sc.FREEZE.read_text(encoding='utf-8'))
    assert hashlib.sha256(sc.PROTOCOL.read_bytes()).hexdigest() == freeze['protocol_sha256']
    for path, digest in protocol['inputs_sha256'].items():
        assert hashlib.sha256((br.ROOT.parent / path).read_bytes()).hexdigest() == digest, path
    return dict(protocol_sha256=freeze['protocol_sha256'])


def window_check(tables):
    data = dict(np.load(sc.WINDOW_TABLES))
    w20 = int(np.flatnonzero(data['windows'] == 20)[0])
    naive_error = float(np.abs(data['naive_rise'][w20] - tables['naive_rise']).max())
    rise_error = float(np.abs(data['rise'][w20] - tables['rise']).max())
    assert naive_error < 1e-6 and rise_error < 1e-6, (naive_error, rise_error)
    assert (np.diff(data['rise'], axis=0) >= 0).all() and (np.diff(data['naive_rise'], axis=0) >= 0).all()
    return dict(w20_max_abs_difference=max(naive_error, rise_error), rise_non_decreasing_in_window=True)


def baseline_check(result):
    e1 = json.loads(st.RESULTS.read_text(encoding='utf-8'))['repeats']
    for r in range(st.SPLITS):
        mine, theirs = result['cells']['baseline'][str(r)], e1[str(r)]['0.005']
        close(mine['choices'], theirs['arms']['certified']['choices'], f'choices {r}')
        close(mine['rules'], theirs['arms']['certified']['rules'], f'rules {r}')
        close(mine['paired'], theirs['arms']['certified']['paired'], f'paired {r}')
        close(mine['oracle'], theirs['oracle'], f'oracle {r}')
    return dict(baseline_equals_e1_certified=True)


def plant_check(tables):
    model = br.success_table(tables)
    for scenario, neutral in (('shift', 0.), ('amplitude', 1.), ('rate', 1.), ('heterogeneity', 0.)):
        assert np.array_equal(sc.plant_success(tables, scenario, neutral), model), scenario
    for scenario, levels in sc.MISMATCH.items():
        previous = model
        for level in levels:
            plant = sc.plant_success(tables, scenario, level)
            if scenario != 'heterogeneity':
                assert (plant <= previous).all() and np.array_equal(plant[0], model[0]), (scenario, level)
                previous = plant
            assert (np.diff(plant.astype(int), axis=-1) >= 0).all(), (scenario, level)
    return dict(neutral_levels_equal_model=True, monotone_in_severity=True, readiness_monotone_in_time=True)


def anchor_check(tables, result):
    stored = json.loads(wm.RESULTS.read_text(encoding='utf-8'))['descriptive']['shift_equivalents']['mixed']
    anchor = sc.anchors(tables)
    assert abs(anchor['crossing_min'] - stored['crossing_min']) < 1e-9
    assert abs(anchor['median_ratio_at_60'] - stored['b3_median_at_wait']) < 1e-9
    assert abs(anchor['measured_ratio'] - stored['measured_median']) < 1e-12
    close(anchor, result['anchors'])
    return dict(crossing_equals_mismatch_study=True, shift_equivalent_min=60. - anchor['crossing_min'])


def leakage_check(tables, result):
    _, _, T = st.roles(0)
    plants = {}
    for scenario, level in sc.scenarios(tables):
        plant = sc.plant_success(tables, scenario, level).copy()
        plant[:, T, :] = ~plant[:, T, :]
        plants[f'{scenario}/{level:g}'] = plant
    rerun = sc.run_mismatch(0, tables=tables, plants=plants)
    for key, block in rerun['scenarios'].items():
        if 'recalibrated' in block:
            close(block['recalibrated']['choices'], result['mismatch']['0']['scenarios'][key]['recalibrated']['choices'], key)
    close(rerun['model_choices'], result['mismatch']['0']['model_choices'])
    return dict(test_outcomes_inverted_choices_unchanged=True)


def recompute_check(tables, result):
    for cell in ('window_10', 'kappa_0.8'):
        close(sc.run_cell(0, cell, tables=tables), result['cells'][cell]['0'], cell)
    close(sc.run_mismatch(0, tables=tables), result['mismatch']['0'], 'mismatch')
    return dict(recomputed=['cell window_10 split 0', 'cell kappa_0.8 split 0', 'mismatch split 0'])


def summary_check(result):
    for cell in sc.TASK_CELLS:
        rows = [result['cells'][cell][str(r)] for r in range(st.SPLITS)]
        block = result['summary']['task'][cell]
        for n in st.RULES:
            assert block['meets_per_history'][n] == sum(all(row['rules'][n][h]['success_rate'] >= .9 for h in br.HISTORIES)
                                                        for row in rows), (cell, n)
        assert block['oracle_feasible'] == sum(all(row['oracle'][h]['success_rate'] >= .9 for h in br.HISTORIES) for row in rows)
        for n in st.ADAPTIVE:
            values = [row['paired'][n]['all']['completion'] for row in rows]
            assert abs(block['paired_completion'][n]['median'] - float(np.median(values))) < 1e-12
            assert result['statements']['E4-S1'][cell]['faster'][n] == sum(v < 0 for v in values)
    for key, block in result['summary']['mismatch'].items():
        for arm in sc.ARMS:
            if arm not in block:
                continue
            rows = [result['mismatch'][str(r)]['scenarios'][key][arm] for r in range(st.SPLITS)]
            for n in st.RULES:
                assert block[arm]['meets_per_history'][n] == sum(
                    all(row['rules'][n][h]['success_rate'] >= .9 for h in br.HISTORIES) for row in rows), (key, arm, n)
    return dict(summary_rederived=True)


def main():
    protocol = sc.load_protocol()
    result = json.loads(sc.RESULTS.read_text(encoding='utf-8'))
    assert result['status'] == 'complete' and result['protocol_sha256'] == hashlib.sha256(sc.PROTOCOL.read_bytes()).hexdigest()
    tables = dict(np.load(br.TABLES))
    checks = dict(freeze=freeze_check(protocol), windows=window_check(tables), baseline=baseline_check(result),
                  plants=plant_check(tables), anchors=anchor_check(tables, result),
                  leakage=leakage_check(tables, result), recompute=recompute_check(tables, result),
                  summary=summary_check(result))
    OUT.write_text(json.dumps(dict(status='passed', checks=checks), indent=2) + '\n', newline='\n')
    print('E4 scope verification: passed')


if __name__ == '__main__':
    main()
