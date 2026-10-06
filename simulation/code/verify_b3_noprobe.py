"""Audit the no-probe control without changing any setting.

Checks that the new no-probe trajectories reproduce the saved history
trajectories, that the new probe rises reproduce the saved table rises,
re-simulates a random subset, and checks that rescoring the version-3
decisions with the original labels reproduces version 3.
"""
import json
import sys

import numpy as np

import b3_model as b3
import b3_noprobe as nb
import b3_readiness as br
import b3_waiting as bw


def consistency_check(tables, arrays):
    grid_index = br.GRID.astype(int)
    noprobe = arrays['noprobe']
    worst_traj = float(np.abs(noprobe[:, :, grid_index] - tables['history_fret'].astype(float)).max())
    worst_rise = float(np.abs(arrays['rise_probe'] - tables['rise'].astype(float)).max())
    assert worst_traj < 1e-5 and worst_rise < 1e-5, (worst_traj, worst_rise)
    return dict(max_abs_noprobe_vs_history=worst_traj, max_abs_probe_rise_vs_tables=worst_rise)


def resimulation_check(protocol, arrays):
    model, posterior = b3.load_model(), b3.load_posterior()
    rng = np.random.default_rng(11)
    worst = 0.
    for plant in rng.choice(len(posterior), 4, replace=False):
        for h, name in enumerate(br.HISTORIES):
            history = [tuple(seg) for seg in protocol['histories'][name]]
            for k in (10, 40, 55):
                probe, noprobe = nb.window_pair(model, posterior[plant], history, protocol['command_ng_per_ml'], float(br.GRID[k]))
                worst = max(worst, abs(float((probe - noprobe).max()) - float(arrays['probe_increment'][h, plant, k])),
                            abs(float(noprobe.max() - noprobe[0]) - float(arrays['rise_without_probe'][h, plant, k])))
    assert worst < 1e-5, worst
    return dict(entries=4 * 2 * 3, max_abs_difference=worst)


def rescoring_check(saved):
    v3 = json.loads(bw.RESULTS.read_text(encoding='utf-8'))['noise']['0.005']['evaluation']
    names = {'current': 'current', 'smoothed': 'smoothed', 'history': 'history', 'fixed': 'fixed_calibrated'}
    worst = max(abs(saved['rescored']['original'][ours][key] - v3[theirs][key])
                for ours, theirs in names.items() for key in ('success_rate', 'mean_completion_min'))
    assert worst < 1e-12, worst
    return dict(max_abs_difference=worst)


def main():
    nb.load_protocol()
    protocol = br.load_protocol()
    tables = dict(np.load(br.TABLES))
    arrays = dict(np.load(nb.ARRAYS))
    saved = json.loads(nb.RESULTS.read_text(encoding='utf-8'))
    report = dict(status='passed', protocol_sha256=nb.protocol_sha256(), consistency=consistency_check(tables, arrays),
                  resimulation=resimulation_check(protocol, arrays), rescoring=rescoring_check(saved),
                  scope='Simulation of the published B3 model; not evidence about measured cells.')
    (br.OUT / 'noprobe_verification_results.json').write_text(json.dumps(report, indent=2) + '\n', newline='\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    sys.exit(main())
