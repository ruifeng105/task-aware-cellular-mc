"""Audit the expanded fresh-test analysis without fitting or retuning.

Checks the frozen protocol hash, the Git blob identity of every expanded
source file, schedules against the authors' XML and simulations, feature
causality on the new blocks, and recomputes every saved score from the
saved ridge coefficients.
"""
import copy
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

import calibrate_fgf2 as c
import expanded_fgf2 as e
import verify_fgf2_pipeline as v

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'public_data/fgf2_author_repository'
MANIFEST = ROOT / 'public_data/fgf2_expanded_source_manifest.json'
B3_SIM = SOURCE / 'Inference_results/Fgf_B3/results/sus_3_20'


def git_blob_sha(raw):
    return hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()


def verify_expanded_sources():
    entries = [m for m in json.loads(MANIFEST.read_text(encoding='utf-8')) if 'path' in m]
    for item in entries:
        raw = (SOURCE / item['path']).read_bytes()
        assert git_blob_sha(raw) == item['blob_sha'], item['path']
    return len(entries)


def xml_schedule_check(blocks):
    """New conditions that the B3 XML defines must match its commands exactly."""
    text = (c.SOURCE / 'FGF2_models/Fgf_B3/config_fgf_sus_3_20.xml').read_text()
    root = ET.fromstring('<config>' + text + '</config>')
    defined = {name for item in root.findall('./inputs/input') for name in item.findtext('experiments').split()}
    in_xml = {key: b for key, b in blocks.items() if key.replace('fgf_', '', 1) in defined}
    checked = v.xml_schedule_check(in_xml)['max_command_errors_ng_ml']
    return dict(checked_against_xml=checked,
                not_defined_in_xml=sorted(set(blocks) - set(in_xml)))


def sp60_timing_check():
    """The authors' simulations of a pulse equal their sustained simulation
    until the pulse ends, then diverge after a short response lag."""
    load = lambda name: c.load_matrix(B3_SIM / name).ravel()
    t = load('sim_times.txt')
    result = {}
    for token in ('2-5ng', '25ng', '250ng'):
        sustained = load(f'sim_sus_{token}_measurements.txt')
        for protocol, end in (('sp_10', 10.), ('sp_60', 60.)):
            difference = np.abs(load(f'sim_{protocol}_{token}_measurements.txt') - sustained)
            assert difference[t <= end].max() <= .001, (protocol, token)
            first = float(t[np.argmax(difference > .005)]) if (difference > .005).any() else None
            result[f'{protocol}_{token}'] = dict(max_difference_until_end=float(difference[t <= end].max()),
                                                 first_time_difference_above_0_005=first)
    for key, row in result.items():
        end = 10. if key.startswith('sp_10') else 60.
        assert row['first_time_difference_above_0_005'] is not None, key
        assert end < row['first_time_difference_above_0_005'] <= end + 20., key
    return dict(pulses=result, scope='Brackets the pulse end between the last equal and first diverging sample; the exact minute is not identified, hence the +/-1 min sensitivity.')


def feature_causality_check(blocks):
    small = {key: dict(b, y=b['y'][:1].copy()) for key, b in blocks.items()}
    changed = copy.deepcopy(small)
    cutoff = 30.
    for b in changed.values():
        mask = b['time'] > cutoff
        b['y'][:, mask] += 10. + np.arange(mask.sum())
    rows_checked = {}
    for horizon in (2., 10.):
        original = c.forecasting_rows(small, horizon)
        perturbed = c.forecasting_rows(changed, horizon)
        before = original.time_min.le(cutoff)
        features = [x for x in original if x != 'target']
        pd.testing.assert_frame_equal(original.loc[before, features], perturbed.loc[before, features])
        rows_checked[str(horizon)] = int(before.sum())
    return dict(past_feature_rows_checked=rows_checked)


def saved_score_check(blocks, protocol):
    """Recompute every saved role score from the saved coefficients."""
    results = json.loads(e.RESULTS.read_text(encoding='utf-8'))
    assert results['protocol_sha256'] == e.protocol_sha256()
    checked = 0
    for horizon, saved in results['horizons'].items():
        rows = c.forecasting_rows(blocks, float(horizon))
        for role, models in saved['roles'].items():
            part = rows[rows.run_id.isin(protocol['roles'][role])]
            for name, metric in models.items():
                model = saved['models'][name]
                prediction = part.current_y.to_numpy() if model is None else c.predict(part, model)
                recomputed = c.run_metric(part, prediction)['run_equal_mse']
                np.testing.assert_allclose(recomputed, metric['run_equal_mse'], rtol=1e-10, atol=1e-14)
                checked += 1
    return dict(role_model_scores_recomputed=checked)


def main():
    protocol = e.load_protocol()
    blocks, inventory = e.load_blocks(protocol)
    report = dict(status='passed', protocol_sha256=e.protocol_sha256(),
                  expanded_source_git_blobs_verified=verify_expanded_sources(),
                  n_conditions=len(blocks), n_cells=sum(len(b['y']) for b in blocks.values()),
                  xml_commands=xml_schedule_check(blocks),
                  sp60_timing=sp60_timing_check(),
                  feature_causality=feature_causality_check(blocks),
                  saved_scores=saved_score_check(blocks, protocol),
                  limitations=['No independent experimental days.',
                               'The 60-min pulse timing is inferred from author simulations, not annotated.'])
    (e.OUT / 'verification_results.json').write_text(json.dumps(report, indent=2) + '\n', newline='\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    sys.exit(main())
