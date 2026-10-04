"""Audit the real-data pilot without fitting or retuning its models.

Checks command schedules against the original author's XML, tests whether
future response changes alter past features, and recomputes saved scores.
"""
import copy
import json
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

import calibrate_fgf2 as c


def xml_schedule_check(blocks):
    text = (c.SOURCE / 'FGF2_models/Fgf_B3/config_fgf_sus_3_20.xml').read_text()
    root = ET.fromstring('<config>' + text + '</config>')
    probe = np.arange(0., 260., .5)
    errors = {}
    for key, block in blocks.items():
        experiment = key.replace('fgf_', '', 1)
        expected = np.zeros_like(probe)
        for item in root.findall('./inputs/input'):
            experiments = item.findtext('experiments').split()
            if experiment not in experiments:
                continue
            index = experiments.index(experiment)

            def value(name):
                values = item.findtext(name).split()
                return float(values[0] if len(values) == 1 else values[index])

            start = value('startingtime')
            duration = value('duration')
            period = value('period')
            strength = value('strength')
            n = min(int(value('numpulses')), int((probe[-1] - start) / period) + 1)
            for k in range(max(0, n)):
                a = start + k * period
                expected += strength * ((probe >= a) & (probe < a + duration))
        actual = c.command(probe, block['protocol'], block['concentration'])
        np.testing.assert_array_equal(actual, expected)
        errors[key] = float(np.max(np.abs(actual - expected)))
    # Total commanded exposure, not molecule number or measured local dose.
    for conc in c.CONCS.values():
        assert c.dose(0., 258., 'fgf_mixed', conc) == 38. * conc
    path = c.DATA / 'fgf_mixed'
    parse = lambda name: [tuple(map(float, line.split('-'))) for line in (path / name).read_text().splitlines() if line.strip()]
    original = parse('pulses.txt')
    truncated = parse('pulses_trunc.txt')
    assert truncated == c.segments('fgf_mixed')
    assert [(a - 42., b - 42.) for a, b in original] == truncated
    return dict(max_command_errors_ng_ml=errors, mixed_exposure_minutes=38.,
                mixed_pulse_annotations_agree_with_xml=True,
                mixed_pulse_intervals_min=truncated,
                exposure_unit='ng min / ml; molecular release requires measured flow')


def feature_causality_check(blocks):
    # One real trajectory per block is sufficient to exercise every protocol.
    small = {key: dict(b, y=b['y'][:1].copy()) for key, b in blocks.items()}
    cutoff = 30.
    changed = copy.deepcopy(small)
    for b in changed.values():
        mask = b['time'] > cutoff
        b['y'][:, mask] += 10. + np.arange(mask.sum())
    checks = {}
    for horizon in (2., 10.):
        original = c.forecasting_rows(small, horizon)
        perturbed = c.forecasting_rows(changed, horizon)
        before = original.time_min.le(cutoff)
        features = [x for x in original if x != 'target']
        pd.testing.assert_frame_equal(original.loc[before, features],
                                      perturbed.loc[before, features])
        assert not np.array_equal(original.loc[before, 'target'],
                                  perturbed.loc[before, 'target'])
        parts = c.validate_split(original, c.SPLIT)
        checks[str(horizon)] = dict(past_feature_rows_checked=int(before.sum()),
                                  future_targets_changed=True,
                                  feature_columns_checked=len(features),
                                  split_condition_counts={k: v.run_id.nunique() for k, v in parts.items()})
    return dict(forecast_horizons_min=checks,
                scope='Feature generation is causal conditional on author-exported normalized inputs. Future commanded input is intentionally available.')


def normalization_check():
    directory = c.DATA / 'fgf_mixed'
    t = c.load_matrix(directory / 'time.txt').ravel()
    tt = c.load_matrix(directory / 'time_trunc.txt').ravel()
    # MATLAB first_pulse_index=22 uses one-based indexing.
    start_index = 21
    np.testing.assert_array_equal(t[start_index:] - t[start_index], tt)
    baseline = (t >= 10.) & (t <= 30.)
    assert t[baseline].max() < t[start_index]
    errors = {}
    for token in c.CONCS:
        raw = c.load_matrix(directory / (token + '_unnormalized.txt'))
        supplied = c.load_matrix(directory / (token + '_trunc.txt'))
        reconstructed = raw[:, start_index:] / np.median(raw[:, baseline], axis=1)[:, None]
        error = float(np.max(np.abs(reconstructed - supplied)))
        # Text exports are rounded independently; exact equality is not expected.
        assert error < .0002
        errors[token] = error
    return dict(exported_origin_original_min=float(t[start_index]),
                normalization_baseline_original_min=[float(t[baseline].min()), float(t[baseline].max())],
                all_normalization_baseline_precedes_exported_origin=True,
                max_reconstruction_error=errors,
                scope='Mixed protocol checked from unnormalized source export and normalize_data.m. Other protocols retain author-preprocessed measurements.')


def saved_score_check():
    result = json.loads((c.OUT / 'calibration_results.json').read_text())
    scores = {}
    for f in result['forecasting']:
        horizon = f['horizon_min']
        rows = pd.read_csv(c.OUT / f'forecasts_{horizon:g}min.csv')
        scores[str(horizon)] = {}
        for name, model in f['models'].items():
            computed = c.run_metric(rows, rows[name].to_numpy())['run_equal_mse']
            np.testing.assert_allclose(computed, model['test']['run_equal_mse'], rtol=1e-10, atol=1e-14)
            scores[str(horizon)][name] = float(np.sqrt(computed))
    return scores


def main():
    verified = c.verify_sources()
    blocks, inventory = c.read_blocks()
    report = dict(status='passed', source_git_blobs_verified=verified,
                  n_cells=sum(x['n_cells'] for x in inventory),
                  n_measurements=sum(x['n_measurements'] for x in inventory),
                  xml_commands=xml_schedule_check(blocks),
                  feature_causality=feature_causality_check(blocks),
                  mixed_normalization=normalization_check(),
                  recomputed_test_rmse=saved_score_check(),
                  limitations=['No assertion of independent experimental days.',
                               'No new live experiment, transport calibration, biochemical parameter inference or policy effect.'])
    target = c.OUT / 'verification_results.json'
    target.write_text(json.dumps(report, indent=2) + '\n', newline='\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
