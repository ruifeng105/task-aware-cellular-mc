"""Next-command forecast errors split by the following pulse (frozen 10-min nested models).

Implements configs/nested_onset_split_protocol.json, frozen before its only
run. In the two mixed Test C conditions the 10-min next-command rows precede
either the 30-min pulse at 24 min (origins 14-22 min, after an initial 3-min
pulse) or the 5-min pulse at 114 min (origins 104-112 min, 60 min after the
30-min pulse). The frozen models are re-applied as in nested_phase_detail.py,
without fitting or selection; the groups recombine to the frozen scores.
Descriptive and post hoc on inspected roles.
"""
import hashlib
import json

import numpy as np

import b3_forecast as bf
import calibrate_fgf2 as c
import expanded_fgf2 as e
import nested_forecast as nf
from history_baselines import run_metric

PROTOCOL = nf.ROOT / 'configs/nested_onset_split_protocol.json'
FREEZE = nf.OUT / 'nested_onset_split_protocol_freeze.json'
RESULTS = nf.OUT / 'onset_split_results.json'
DETAIL = nf.OUT / 'phase_detail.json'
HORIZON = 10.
GROUPS = {24.: ('before_30min_pulse', (14., 16., 18., 20., 22.)),
          114.: ('before_5min_probe', (104., 106., 108., 110., 112.))}
NAMES = ('P',) + nf.MODELS
BOOTSTRAP, BOOT_SEED = 2000, 20261011
LONG_CONDITION = 'fgf_mixed_25ng'
SHARE_THRESHOLD = .75


def protocol_sha256():
    return hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()


def load_protocol():
    if protocol_sha256() != json.loads(FREEZE.read_text(encoding='utf-8'))['protocol_sha256']:
        raise SystemExit('Protocol differs from its freeze record; refusing to run.')
    protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    for rel, expected in protocol['inputs_sha256'].items():
        if hashlib.sha256((nf.ROOT.parent / rel).read_bytes()).hexdigest() != expected:
            raise SystemExit(f'Input {rel} differs from the hash pinned in the protocol.')
    return protocol


def following_onset(origin, horizon, protocol):
    """Start of the first pulse in (origin, origin + horizon], else None."""
    starts = [a for a, _ in c.segments(protocol, until=origin + horizon + 1.) if origin < a <= origin + horizon]
    return min(starts) if starts else None


def next_command_rows(data, roles):
    keys = [k for r in e.NEW_TEST_ROLES for k in roles[r]]
    test = data[data.run_id.isin(keys)].copy()
    test['phase'] = [nf.phase(t, HORIZON, p) for t, p in zip(test.time_min, test.history_id)]
    test = test[(test.phase == 'next_command').to_numpy()].copy()
    test['onset'] = [following_onset(t, HORIZON, p) for t, p in zip(test.time_min, test.history_id)]
    if set(test.onset) != set(GROUPS):
        raise SystemExit(f'unexpected next-command onsets {sorted(set(test.onset))}')
    return test


def cell_table(part, preds, mask):
    """Per cell: rows, sum of squared errors and of signed errors per model."""
    cells = part.cell_id.to_numpy()
    ids = np.unique(cells)
    index = np.searchsorted(ids, cells)
    target = part.target.to_numpy()
    out = dict(ids=ids, rows=np.bincount(index, minlength=ids.size).astype(float))
    for m in NAMES:
        err = preds[m][mask] - target
        out[f'sse_{m}'] = np.bincount(index, weights=err ** 2, minlength=ids.size)
        out[f'se_{m}'] = np.bincount(index, weights=err, minlength=ids.size)
    return out


def bootstrap(table, rng):
    """95% percentile intervals over cell resamples for MSE_M - MSE_O and the signed errors of O and M."""
    n = table['ids'].size
    draws = rng.integers(0, n, (BOOTSTRAP, n))
    rows = table['rows'][draws].sum(1)
    stat = {'mse_M_minus_O': (table['sse_M'][draws].sum(1) - table['sse_O'][draws].sum(1)) / rows,
            'signed_O': table['se_O'][draws].sum(1) / rows, 'signed_M': table['se_M'][draws].sum(1) / rows}
    return {k: [float(x) for x in np.percentile(v, [2.5, 97.5])] for k, v in stat.items()}


def split(data, roles, models):
    test = next_command_rows(data, roles)
    preds = nf.predictions(test, models, names=nf.MODELS)
    rng = np.random.default_rng(BOOT_SEED)
    out = {}
    for onset, (group, origins) in GROUPS.items():
        mask = (test.onset == onset).to_numpy()
        part = test[mask]
        per_model = {m: run_metric(part, yp[mask])['per_run'] for m, yp in preds.items()}
        conditions = {}
        for key in sorted(part.run_id.unique()):
            sel = (part.run_id == key).to_numpy()
            rows = part[sel]
            if tuple(sorted(rows.time_min.unique())) != origins:
                raise SystemExit(f'{group} {key}: origins {sorted(rows.time_min.unique())}')
            if len(rows) != len(origins) * rows.cell_id.nunique():
                raise SystemExit(f'{group} {key}: rows {len(rows)} != 5 x cells')
            local = {m: yp[mask][sel] for m, yp in preds.items()}
            target = rows.target.to_numpy()
            table = cell_table(rows, local, np.ones(len(rows), dtype=bool))
            rmse = {m: float(np.sqrt(per_model[m][key]['mse'])) for m in NAMES}
            mse = {m: per_model[m][key]['mse'] for m in NAMES}
            per_cell_m = table['sse_M'] / table['rows']
            per_cell_o = table['sse_O'] / table['rows']
            conditions[key] = dict(rows=int(len(rows)), cells=int(rows.cell_id.nunique()), origins=list(origins),
                                   rmse=rmse, relative_change_M_vs_O=rmse['M'] / rmse['O'] - 1,
                                   mse_M_minus_O=float(mse['M'] - mse['O']),
                                   signed_error={m: float(np.mean(local[m] - target)) for m in NAMES},
                                   mean_b3_delta=float(rows.b3_delta.mean()),
                                   mean_realized_change=float(np.mean(target - rows.current_y.to_numpy())),
                                   cells_M_worse_than_O=int((per_cell_m > per_cell_o).sum()),
                                   bootstrap_95=bootstrap(table, rng))
        pooled = {m: float(np.sqrt(np.mean([row['rmse'][m] ** 2 for row in conditions.values()]))) for m in NAMES}
        out[group] = dict(onset_min=onset, conditions=conditions, pooled=pooled,
                          relative_change_M_vs_O=pooled['M'] / pooled['O'] - 1)
    return out


def shares(groups):
    """Each group's share of the window's M - O MSE increase, per condition and pooled (five rows per cell each)."""
    keys = sorted(next(iter(groups.values()))['conditions'])
    out = {}
    for key in keys + ['pooled']:
        if key == 'pooled':
            diff = {g: np.mean([b['conditions'][k]['mse_M_minus_O'] for k in keys]) for g, b in groups.items()}
        else:
            diff = {g: b['conditions'][key]['mse_M_minus_O'] for g, b in groups.items()}
        window = sum(diff.values()) / 2
        out[key] = {g: float(d / (2 * window)) if window != 0 else None for g, d in diff.items()}
    return out


def reading(groups, share):
    late = groups['before_5min_probe']['conditions'][LONG_CONDITION]
    early = groups['before_30min_pulse']['conditions'][LONG_CONDITION]
    located = (share[LONG_CONDITION]['before_5min_probe'] is not None
               and share[LONG_CONDITION]['before_5min_probe'] >= SHARE_THRESHOLD
               and late['bootstrap_95']['mse_M_minus_O'][0] > 0)
    overestimates = late['signed_error']['M'] > 0 and late['signed_error']['M'] > late['signed_error']['O']
    underestimates_early = early['signed_error']['M'] < 0
    return dict(located_before_probe_after_long_command=bool(located),
                consistent_with_overestimated_recovery=bool(overestimates),
                negative_M_error_before_30min_pulse=bool(underestimates_early),
                statement=('located before the probe that follows the long command' if located
                           else 'the split does not locate the error'))


def combined_check(groups):
    """Row-weighted MSE over the two groups must equal the frozen per-condition next-command RMSE."""
    detail = json.loads(DETAIL.read_text(encoding='utf-8'))['phases']['next_command']
    worst = 0.
    for key, frozen in detail.items():
        rows = [groups[g]['conditions'][key] for g, _ in GROUPS.values()]
        n = sum(row['rows'] for row in rows)
        assert n == frozen['rows'], (key, n, frozen['rows'])
        for m in NAMES:
            mse = sum(row['rows'] * row['rmse'][m] ** 2 for row in rows) / n
            worst = max(worst, abs(np.sqrt(mse) - frozen['rmse'][m]))
    assert worst < 1e-12, worst
    return worst


def compute():
    protocol, roles, blocks, population = nf.load_inputs()
    analysis = json.loads(nf.RESULTS.read_text(encoding='utf-8'))['main']['10']
    rows = bf.forecast_rows(blocks, population, HORIZON, [nf.sigma_value(analysis['sigma'])])
    data = rows.assign(b3_delta=rows[f'b3_delta_{analysis["sigma"]}'])
    return split(data, roles, analysis['models'])


def main():
    load_protocol()
    groups = compute()
    worst = combined_check(groups)
    share = shares(groups)
    result = dict(status='complete', protocol_sha256=protocol_sha256(),
                  frozen_utc=json.loads(FREEZE.read_text(encoding='utf-8'))['frozen_utc'],
                  source_sha256=hashlib.sha256(nf.RESULTS.read_bytes()).hexdigest(),
                  horizon_min=HORIZON, groups=groups, share_of_increase=share, reading=reading(groups, share),
                  combined_max_abs_difference=worst,
                  scope='Descriptive breakdown of the frozen 10-min nested models; post hoc on inspected roles; '
                        'five origins per group and condition; groups also differ in the following pulse and history length.')
    RESULTS.write_text(json.dumps(result, indent=2) + '\n', newline='\n')
    for group, block in groups.items():
        print(group, 'pooled M/O-1 %.1f%%' % (100 * block['relative_change_M_vs_O']))
        for key, row in block['conditions'].items():
            print('   ', key, row['rows'], {m: round(v, 5) for m, v in row['rmse'].items()},
                  'M/O-1 %.1f%%' % (100 * row['relative_change_M_vs_O']), 'signed', {m: round(v, 4) for m, v in row['signed_error'].items()},
                  'b3_delta', round(row['mean_b3_delta'], 4), 'real', round(row['mean_realized_change'], 4),
                  'cells M>O', row['cells_M_worse_than_O'], 'boot', row['bootstrap_95'])
    print('shares', share)
    print('reading', result['reading'])


if __name__ == '__main__':
    main()
