"""Audit the condition map at calibration targets 0.92 and 0.94.

Implements the checks of configs/b3_delay_map_target_protocol.json: recorded
reproduction of the frozen 0.90 map; equality with the frontier study at
target 0.94 in the no-delay, 2-min cells; fixed choices identical across the
cells of a split; feasibility of flagged choices; leakage; two jobs
recomputed; classes re-derived from the saved splits.
"""
import json
import sys

import numpy as np

import b3_delay_map as dm
import b3_delay_map_target as dt
import b3_readiness as br
import b3_waiting_frontier as wf
import b3_waiting_robustness as rb

TOL = 1e-12


def frontier_check(saved):
    frontier = json.loads(wf.RESULTS.read_text(encoding='utf-8'))['repeats']
    compared = 0
    for noise in ('0.005', '0.01'):
        key = dm.cell_key(float(noise), 2, 0)
        for r in range(rb.REPEATS):
            cell = saved['splits']['0.94'][key][str(r)]['rules']
            ref = frontier[str(r)][noise]['cells']['0.94']['rules']
            for mine, theirs in (('b3_predictive', 'history'), ('b3_nowcast', 'history'), ('fixed', 'fixed')):
                for part, row in cell[mine].items():
                    for k, v in row.items():
                        assert abs(v - ref[theirs][part][k]) <= TOL, (noise, r, mine, part, k)
            compared += 1
    return dict(splits_compared=compared)


def fixed_check(saved):
    for t, cells in saved['splits'].items():
        for r in range(rb.REPEATS):
            choices = {json.dumps(rows[str(r)]['choices']['fixed'], sort_keys=True) for rows in cells.values()}
            assert len(choices) == 1, (t, r)
    return dict(identical_across_cells=True)


def feasibility_check(saved):
    count = 0
    for t, cells in saved['splits'].items():
        for key, rows in cells.items():
            for row in rows.values():
                for name, per_history in row['choices'].items():
                    for choice in per_history.values():
                        if choice['target_reached']:
                            assert choice['calibration']['success_rate'] >= float(t) - TOL, (t, key, name)
                            count += 1
    return dict(feasible_choices_checked=count)


def leakage_check(saved, r=6, noise=.01, interval=6):
    tables, success, cal, ev, observed = dt.job_parts(r, noise, interval)
    flipped = success.copy()
    flipped[:, ev, :] = ~flipped[:, ev, :]
    options = dm.rule_options(tables, flipped, observed, noise, interval, cal, cal)
    for d in dm.DELAYS:
        key = dm.cell_key(noise, interval, d)
        choices = {name: {hist: wf.choose(options[d][name], h, .94) for h, hist in enumerate(br.HISTORIES)}
                   for name in dm.RULES}
        assert json.loads(json.dumps(choices)) == saved['splits']['0.94'][key][str(r)]['choices'], key
    return dict(r=r, noise=noise, interval=interval, evaluation_outcomes_inverted=True, choices_unchanged=True)


def recompute_check(saved, jobs=((4, .005, 10), (11, .01, 2))):
    frozen = json.loads(dm.RESULTS.read_text(encoding='utf-8'))['splits']
    for r, noise, interval in jobs:
        fresh = dt.run_job(r, noise, interval)
        for d in map(str, dm.DELAYS):
            key = dm.cell_key(noise, interval, int(d))
            for t in dt.REPORTED:
                assert fresh['cells'][t][d] == saved['splits'][t][key][str(r)], (t, key, r)
            assert fresh['cells']['0.90'][d]['choices'] == frozen[key][str(r)]['choices'], (key, r)
    return dict(jobs=[list(j) for j in jobs], identical=True)


def class_check(saved):
    for t, cells in saved['splits'].items():
        for key, rows in cells.items():
            assert dt.summarize([rows[str(r)] for r in range(rb.REPEATS)]) == saved['summary'][t][key], (t, key)
    return dict(cells=sum(len(c) for c in saved['splits'].values()))


def main():
    dt.load_protocol()
    saved = json.loads(dt.RESULTS.read_text(encoding='utf-8'))
    assert saved['reproduction_at_090_max_abs_difference'] < TOL
    report = dict(status='passed', protocol_sha256=dt.protocol_sha256(),
                  reproduction_at_090=saved['reproduction_at_090_max_abs_difference'],
                  frontier=frontier_check(saved), fixed=fixed_check(saved), feasibility=feasibility_check(saved),
                  leakage=leakage_check(saved), recomputation=recompute_check(saved), classes=class_check(saved),
                  scope='Simulation of the published B3 model; not evidence about measured cells.')
    (br.OUT / 'delay_map_target_verification_results.json').write_text(json.dumps(report, indent=2) + '\n', newline='\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    sys.exit(main())
