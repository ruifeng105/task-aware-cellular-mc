"""Audit the in-silico readiness study without changing any setting.

Checks the frozen protocol hash, the decision rules on synthetic inputs, that
the history policy only uses observations up to the decision time, re-simulates
a random subset of saved table entries, recomputes every decision and
summary metric from the saved tables with the registered seeds, and checks the
Fig. 2 example against its selection rule and the saved tables.
"""
import json
import sys

import numpy as np
import pandas as pd

import b3_model as b3
import b3_readiness as br


def rule_checks():
    assert br.first_crossing(np.array([.1, .95, .99]), .9) == 1
    assert br.first_crossing(np.array([.1, .2, .3]), .9) == 2, 'no crossing must fall back to the deadline'
    assert br.half_decay_index(np.array([1.08, 1.09, 1.06, 1.044, 1.04])) == 3
    assert br.half_decay_index(np.array([1.08, 1.09, 1.08])) == 2, 'no half decay must fall back to the deadline'
    rng = np.random.default_rng(1)
    trajectories = 1 + .1 * rng.random((30, 12))
    assert br.success_table(dict(rise=np.array([[[.5, .49]]]), naive_rise=np.array([1.]))).tolist() == [[[True, False]]]
    success = rng.random((30, 12)) > .5
    observed = trajectories[3] + .001
    base = br.history_probability(observed, trajectories, success, plant=3, sigma=.005)
    changed = observed.copy()
    changed[6:] += .5
    later = br.history_probability(changed, trajectories, success, plant=3, sigma=.005)
    assert np.allclose(base[:6], later[:6]), 'history policy used future observations'
    return dict(first_crossing=True, deadline_fallback=True, threshold_rule=True, past_only=True)


def resimulation_check(tables, protocol):
    model, posterior = b3.load_model(), b3.load_posterior()
    rng = np.random.default_rng(7)
    worst = 0.
    for plant in rng.choice(len(posterior), 6, replace=False):
        for h, name in enumerate(br.HISTORIES):
            history = [tuple(seg) for seg in protocol['histories'][name]]
            fret = model.fret_from_states(posterior[plant], model.simulate_schedule_states(
                posterior[plant], history, protocol['command_ng_per_ml'], br.GRID))
            worst = max(worst, float(np.abs(fret - tables['history_fret'][h, plant]).max()))
            for k in (0, 10, 30, 60):
                rise = br.probe_rise(model, posterior[plant], history, protocol['command_ng_per_ml'], br.GRID[k])
                worst = max(worst, abs(rise - float(tables['rise'][h, plant, k])))
    assert worst < 1e-5, worst
    return dict(entries_resimulated=6 * 2 * 5, max_abs_difference=worst)


def version1_check(tables):
    """Version 1 (task 0.8, baseline band 0.01) was infeasible for every sample."""
    ratio = tables['rise'].astype(float) / tables['naive_rise'].astype(float)[None, :, None]
    near_baseline = int((np.abs(tables['history_fret'].astype(float) - 1.) <= .01).sum())
    assert ratio.max() < .8 and near_baseline == 0, (ratio.max(), near_baseline)
    return dict(max_recovered_fraction=float(ratio.max()), states_within_0_01_of_baseline=near_baseline)


def recomputation_check(tables):
    saved = json.loads(br.RESULTS.read_text(encoding='utf-8'))
    decisions = pd.read_csv(br.DECISIONS)
    recomputed = br.decide_all(tables)
    pd.testing.assert_frame_equal(recomputed.reset_index(drop=True), decisions.reset_index(drop=True), check_dtype=False)
    summary = br.summarize(tables, recomputed)
    for policy, row in saved['policies'].items():
        for key in ('success_rate', 'mean_completion_min', 'mean_wait_min'):
            np.testing.assert_allclose(summary['policies'][policy][key], row[key], rtol=1e-12)
    return dict(decisions_recomputed=len(decisions), policies_checked=len(saved['policies']))


def example_check(tables):
    """The Fig. 2 sample follows its selection rule and matches the saved tables and decisions."""
    import b3_readiness_example as ex
    saved = json.loads(ex.EXAMPLE.read_text(encoding='utf-8'))
    decisions = pd.read_csv(br.DECISIONS)
    plant, group, later = ex.select(decisions)
    assert (saved['plant'], saved['group_size'], saved['current_later']) == (plant, group, later)
    h = br.HISTORIES.index(ex.HISTORY)
    rows = decisions[(decisions.history == ex.HISTORY) & (decisions.plant == plant)].set_index('policy')
    for name, policy in (('current', 'readiness_current'), ('history', 'readiness_history')):
        branch = saved['branches'][name]
        assert branch['wait_min'] == rows.wait_min[policy] and branch['success'] == bool(rows.success[policy])
        assert br.GRID[br.first_crossing(np.array(saved['probability'][name]), br.Q)] == branch['wait_min']
    np.testing.assert_allclose(saved['observed'], br.observations(tables)[h, plant], atol=1e-6)
    worst = 0.
    for branch in saved['branches'].values():
        k = int(np.flatnonzero(br.GRID == branch['wait_min'])[0])
        worst = max(worst, abs(branch['rise'] - float(tables['rise'][h, plant, k])))
        assert branch['success'] == bool(br.success_table(tables)[h, plant, k])
    assert worst < 1e-5, worst
    return dict(plant=plant, group_size=group, max_abs_rise_difference=worst)


def main():
    protocol = br.load_protocol()
    tables = dict(np.load(br.TABLES))
    report = dict(status='passed', protocol_sha256=br.protocol_sha256(), rules=rule_checks(),
                  resimulation=resimulation_check(tables, protocol), version1=version1_check(tables),
                  recomputation=recomputation_check(tables), example=example_check(tables),
                  scope='Simulation of the published B3 model; not evidence about measured cells.')
    (br.OUT / 'readiness_verification_results.json').write_text(json.dumps(report, indent=2) + '\n', newline='\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    sys.exit(main())
