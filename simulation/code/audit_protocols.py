"""Trace every stimulation schedule used by the analyses back to the authors' files (revision P0).

For each protocol this checks that the commanded pulses in calibrate_fgf2.segments equal the authors'
schedules (the B3 XML inputs and the simulation summary) and, for the mixed protocol, the recorded pulse file
shifted by the truncation origin of normalize_data.m; that the truncated, normalized measurements are the raw
records cut at that origin; and how far the reporter moves before the first pulse, so that an earlier,
unrecorded stimulation would be visible. Deterministic checks; any mismatch raises.
Usage: python audit_protocols.py
"""
import json
import re
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

import calibrate_fgf2 as c

ROOT = Path(__file__).resolve().parents[1]
AUTHOR = ROOT / 'public_data/fgf2_author_repository'
DATA = AUTHOR / 'FGF2_models/data'
XML = AUTHOR / 'FGF2_models/Fgf_B3/config_fgf_sus_3_20.xml'
SUMMARY = AUTHOR / 'Inference_results/Fgf_B3/results/sus_3_20/sim_post_model_summary.txt'
OUT = ROOT / 'results/audit/protocol_trace.json'
DOSES = ('0-25ng', '2-5ng', '25ng', '250ng')
FOLDERS = {'fgf_sus': 'sus', 'fgf_3_20': '3_20', 'fgf_sp_5': 'sp_5', 'fgf_sp_10': 'sp_10', 'fgf_sp_60': 'sp_60',
           'fgf_mixed': 'mixed'}
HORIZON = 300.
TEXT_PRECISION = 1e-3  # the author files are written by dlmwrite with 5 significant digits


def pulse_file(path):
    return [tuple(float(v) for v in line.split('-')) for line in path.read_text().split() if line.strip()]


def xml_schedules():
    """{experiment: [(start, end, strength), ...]} from the XML <input> blocks (pulse trains expanded to HORIZON)."""
    out = {}
    # The author file holds several top-level elements; parse only its <inputs> element.
    inputs = re.search(r'<inputs>.*?</inputs>', XML.read_text(), flags=re.S).group(0)
    for block in ET.fromstring(inputs):
        names = block.find('experiments').text.split()
        fields = {key: [float(v) for v in block.find(key).text.split()]
                  for key in ('period', 'strength', 'duration', 'numpulses')}
        start = float(block.find('startingtime').text)
        for i, name in enumerate(names):
            period, duration = fields['period'][i], fields['duration'][i]
            for k in range(int(fields['numpulses'][i])):
                onset = start + k * period
                if onset > HORIZON:
                    break
                out.setdefault(name, []).append((onset, min(onset + duration, 1e9), fields['strength'][i]))
    return {k: sorted(v) for k, v in out.items()}


def summary_schedules():
    """{experiment: [(start, end, strength), ...]} for fgf_input from the authors' simulation summary."""
    out, current = {}, None
    pattern = re.compile(r'pulses of strength (\S+) of parameter fgf_input for (\S+) time units, every (\S+), '
                         r'starting from (\S+)')
    for line in SUMMARY.read_text().splitlines():
        head = re.match(r'Experiment (\S+):', line.strip())
        if head:
            current = head.group(1)
            continue
        match = pattern.search(line)
        if match and current:
            strength, duration, period, start = (float(v) for v in match.groups())
            onset = start
            while onset <= HORIZON:
                out.setdefault(current, []).append((onset, min(onset + duration, 1e9), strength))
                onset += period
    return {k: sorted(v) for k, v in out.items()}


def dose_value(token):
    return float(token.replace('ng', '').replace('-', '.'))


def ours(protocol, dose):
    return [(a, min(b, 1e9), dose) for a, b in c.segments(protocol, until=HORIZON)]


def same(a, b):
    return len(a) == len(b) and all(np.allclose(x[:2], y[:2]) and np.isclose(x[2], y[2]) for x, y in zip(a, b))


def clipped(schedule):
    """Sustained inputs end at different large numbers in different files; compare up to the horizon."""
    return [(a, min(b, HORIZON), s) for a, b, s in schedule]


def pre_onset(normalized, raw_time, onset_raw):
    """Median over cells of the largest |y - 1| before the first pulse, and of the largest rise within 20 min after it."""
    before = raw_time < onset_raw
    after = (raw_time > onset_raw) & (raw_time <= onset_raw + 20.)
    return dict(frames_before=int(before.sum()),
                median_max_abs_deviation_before=float(np.nanmedian(np.nanmax(np.abs(normalized[:, before] - 1.), 1))),
                median_max_rise_first_20_min=float(np.nanmedian(np.nanmax(normalized[:, after] - 1., 1))))


def mixed_trace():
    raw_time = np.loadtxt(DATA / 'fgf_mixed/time.txt')
    trunc_time = np.loadtxt(DATA / 'fgf_mixed/time_trunc.txt')
    script = (DATA / 'fgf_mixed/normalize_data.m').read_text()
    first = int(re.search(r'first_pulse_index\s*=\s*(\d+)', script).group(1))
    origin = float(raw_time[first - 1])                      # MATLAB indices start at 1
    assert np.allclose(raw_time[first - 1:] - origin, trunc_time), 'time_trunc is not time cut at the origin'
    recorded = pulse_file(DATA / 'fgf_mixed/pulses.txt')
    truncated = pulse_file(DATA / 'fgf_mixed/pulses_trunc.txt')
    shifted = [(a - origin, b - origin) for a, b in recorded]
    assert np.allclose(shifted, truncated), (shifted, truncated)
    assert np.allclose(truncated, c.segments('fgf_mixed')), (truncated, c.segments('fgf_mixed'))
    doses = {}
    for token in DOSES:
        raw = c.load_matrix(DATA / f'fgf_mixed/{token}_unnormalized.txt')
        trunc = c.load_matrix(DATA / f'fgf_mixed/{token}_trunc.txt')
        baseline = np.median(raw[:, (raw_time >= 10.) & (raw_time <= 30.)], axis=1)
        normalized = raw / baseline[:, None]
        error = float(np.nanmax(np.abs(normalized[:, first - 1:] - trunc)))
        assert error < TEXT_PRECISION, (token, error)
        doses[token] = dict(cells=int(raw.shape[0]), max_abs_difference_to_trunc=error,
                            **pre_onset(normalized, raw_time, recorded[0][0]))
    return dict(recorded_pulses_raw_min=recorded, truncation_first_index_matlab=first, origin_raw_min=origin,
                recorded_minus_origin=shifted, pulses_trunc=truncated, model_segments=c.segments('fgf_mixed'),
                normalization='each cell divided by its median over raw 10-30 min (normalize_data.m); '
                              'the script names 250ng but the same cut reproduces every dose',
                doses=doses)


def other_trace(protocol):
    folder = DATA / protocol
    raw_time, trunc_time = np.loadtxt(folder / 'time.txt'), np.loadtxt(folder / 'time_trunc.txt')
    cut = raw_time.size - trunc_time.size
    origin = float(raw_time[cut] - trunc_time[0])
    assert np.allclose(raw_time[cut:] - origin, trunc_time)
    doses = {}
    for token in DOSES:
        if not (folder / f'{token}.txt').exists():
            continue
        full, trunc = c.load_matrix(folder / f'{token}.txt'), c.load_matrix(folder / f'{token}_trunc.txt')
        error = float(np.nanmax(np.abs(full[:, cut:] - trunc)))
        assert error < TEXT_PRECISION, (protocol, token, error)
        doses[token] = dict(cells=int(full.shape[0]), max_abs_difference_to_trunc=error,
                            **pre_onset(full, raw_time, origin))
    return dict(raw_frames=int(raw_time.size), truncated_frames=int(trunc_time.size), frames_cut=int(cut),
                origin_raw_min=origin, first_truncated_frame_min=float(trunc_time[0]), doses=doses)


def main():
    xml, summary = xml_schedules(), summary_schedules()
    rows = []
    for protocol, short in FOLDERS.items():
        for token in DOSES:
            name = f'{short}_{token}'
            if name not in xml and name not in summary:
                continue
            mine = ours(protocol, dose_value(token))
            row = dict(experiment=name, model_segments=[list(s[:2]) for s in clipped(mine)[:6]],
                       pulses_listed=min(len(mine), 6), dose_ng_per_ml=dose_value(token))
            for label, source in (('xml', xml), ('summary', summary)):
                if name in source:
                    agree = same(clipped(source[name])[:len(mine)], clipped(mine))
                    assert agree, (label, name, source[name][:4], mine[:4])
                    row[f'matches_{label}'] = True
                else:
                    row[f'matches_{label}'] = None
            rows.append(row)
    result = dict(
        status='passed',
        sources=dict(xml=XML.relative_to(ROOT).as_posix(), summary=SUMMARY.relative_to(ROOT).as_posix(),
                     mixed_pulse_record='public_data/fgf2_author_repository/FGF2_models/data/fgf_mixed/pulses.txt',
                     time_unit='min'),
        schedules=rows, mixed=mixed_trace(),
        truncation={p: other_trace(p) for p in FOLDERS if p != 'fgf_mixed'},
        notes=['Commanded concentrations, not measured local concentrations; only the mixed protocol has a recorded '
               'pulse file.',
               'fgf_sp_60 has no XML input; its [0, 60) pulse is listed in the authors\' simulation summary. Its '
               'truncated frames start at 0 min whereas fgf_sp_5 and fgf_sp_10 start at 1 min after the same '
               'raw cut, so its onset is uncertain by 1 min (covered by the registered +-1 min timing sensitivity).',
               'No pulse precedes the recorded first pulse of the mixed protocol; the reporter before it is '
               'summarized in mixed/doses/*/median_max_abs_deviation_before.'])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + '\n', newline='\n')
    print('protocol trace passed;', len(rows), 'experiments matched;',
          'mixed origin', result['mixed']['origin_raw_min'], 'min;',
          {k: (round(v['median_max_abs_deviation_before'], 4), round(v['median_max_rise_first_20_min'], 4))
           for k, v in result['mixed']['doses'].items()})
    for p, t in result['truncation'].items():
        print(p, 'origin', t['origin_raw_min'], 'first frame', t['first_truncated_frame_min'],
              {k: (round(v['median_max_abs_deviation_before'], 4), round(v['median_max_rise_first_20_min'], 4))
               for k, v in t['doses'].items()})


if __name__ == '__main__':
    main()
