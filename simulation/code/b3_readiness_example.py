"""Single-sample illustration of readiness-aware waiting (manuscript Fig. 2).

Selects one 30-min-history sample from the saved readiness decisions by a fixed
rule and recomputes its observations, both success-probability estimates and
the probe responses at three waits. No setting of the readiness study changes;
the aggregate results come from b3_readiness.py. Simulation of the published
B3 model, not evidence about measured cells.
"""
import json

import numpy as np
import pandas as pd

import b3_model as b3
import b3_readiness as br

EXAMPLE = br.OUT / 'readiness_example.json'
HISTORY = 'long'
FIXED_WAIT = 48.  # largest registered candidate wait
RULE = ('30-min history; among samples where the current-only policy probed earlier than the history-aware '
        'policy and failed while the history-aware policy succeeded, the one whose history-aware wait is closest '
        'to the group median (ties: current-only wait closest to its median, then lowest index).')


def select(decisions):
    """Selected plant, size of the eligible group, and how many samples the current-only policy probed later."""
    table = decisions[decisions.history == HISTORY].pivot(index='plant', columns='policy', values=['wait_min', 'success'])
    wait, ok = table['wait_min'], table['success']
    eligible = ((wait.readiness_current < wait.readiness_history) & (ok.readiness_current == 0)
                & (ok.readiness_history == 1))
    group = wait[eligible]
    key = pd.DataFrame({'history': (group.readiness_history - group.readiness_history.median()).abs(),
                        'current': (group.readiness_current - group.readiness_current.median()).abs(),
                        'plant': group.index}).reset_index(drop=True)
    plant = int(key.sort_values(['history', 'current', 'plant']).plant.iloc[0])
    return plant, int(eligible.sum()), int((wait.readiness_current > wait.readiness_history).sum())


def rounded(values):
    return [round(float(v), 7) for v in values]


def main():
    protocol = br.load_protocol()
    tables = dict(np.load(br.TABLES))
    decisions = pd.read_csv(br.DECISIONS)
    plant, group, later = select(decisions)
    h = br.HISTORIES.index(HISTORY)
    success = br.success_table(tables)
    observed = br.observations(tables)[h, plant]
    probability = dict(
        history=br.history_probability(observed, tables['history_fret'][h].astype(float), success[h], plant, br.KERNEL_SD),
        current=br.CurrentReadiness(tables['history_fret'], success, br.KERNEL_SD).probability(observed, plant))
    waits = dict(fixed=FIXED_WAIT, **{name: float(br.GRID[br.first_crossing(p, br.Q)]) for name, p in probability.items()})

    model, row = b3.load_model(), b3.load_posterior()[plant]
    history, dose = [tuple(s) for s in protocol['histories'][HISTORY]], protocol['command_ng_per_ml']
    times = np.arange(history[0][0], br.GRID[-1] + 1., 1.)
    reporter = model.fret_from_states(row, model.simulate_schedule_states(row, history, dose, times))
    on_grid = np.isin(times, br.GRID)
    assert np.abs(reporter[on_grid] - tables['history_fret'][h, plant]).max() < 1e-5
    required = float(br.KAPPA * tables['naive_rise'][plant])
    branches = {}
    for name, wait in waits.items():
        window = np.arange(wait, wait + br.WINDOW_MIN + 1., 1.)
        fret = model.fret_from_states(row, model.simulate_schedule_states(
            row, history + [(wait, wait + br.PROBE_MIN)], dose, window))
        k = int(np.flatnonzero(br.GRID == wait)[0])
        branches[name] = dict(wait_min=wait, times_min=window.tolist(), fret=rounded(fret),
                              rise=float(fret.max() - fret[0]), success=bool(success[h, plant, k]))
    result = dict(rule=RULE, history=HISTORY, command_ng_per_ml=dose, plant=plant, group_size=group,
                  current_later=later, samples=int(tables['naive_rise'].size), q=br.Q, kappa=br.KAPPA,
                  required_rise=required, ready=success[h, plant].astype(int).tolist(),
                  grid_min=br.GRID.tolist(), observed=rounded(observed),
                  probability={name: rounded(p) for name, p in probability.items()},
                  reporter=dict(times_min=times.tolist(), fret=rounded(reporter)), branches=branches,
                  scope='Illustration on one posterior sample of the published B3 model; not evidence about measured cells.')
    EXAMPLE.write_text(json.dumps(result, indent=2) + '\n', newline='\n')
    print(json.dumps(dict(plant=plant, group_size=group, current_later=later,
                          waits=waits, success={n: b['success'] for n, b in branches.items()}), indent=2))


if __name__ == '__main__':
    main()
