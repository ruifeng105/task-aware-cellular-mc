"""Checks for the pilot and expanded pipelines; run after a reproduction.

Usage: python simulation/code/pipeline_checks.py
Exits nonzero with a list of failures.
"""
from pathlib import Path
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'simulation/code'))
import calibrate_fgf2 as c

TEXT_SUFFIXES = {'.json', '.csv', '.tex', '.log', '.txt'}
GENERATED_DIRS = ('simulation/results', 'simulation/configs', 'paper/tables', 'reproduction', 'validation_kit')
KIT = ROOT / 'validation_kit'
# Received validation-kit files changed during integration (LF-only writers on every OS).
KIT_EDITED = {'validation_kit/scripts/design_candidates.py', 'validation_kit/scripts/sample_size.py',
              'validation_kit/scripts/audit_registration.py', 'validation_kit/scripts/audit_data_contract.py',
              'validation_kit/scripts/verify_planning_tools.py'}
GENERATED_FILES = ('paper/assets_manifest.json',)


def generated_text_files():
    files = [ROOT / f for f in GENERATED_FILES]
    for directory in GENERATED_DIRS:
        files += [p for p in (ROOT / directory).rglob('*') if p.suffix in TEXT_SUFFIXES]
    return [p for p in files if p.is_file()]


def check_outputs_lf():
    """Generated text must be byte-identical across platforms: LF only."""
    return [f'CR in {p.relative_to(ROOT)}' for p in generated_text_files() if b'\r' in p.read_bytes()]


def check_schedules():
    """Pilot schedules unchanged; new single pulses start at the export origin."""
    expected = {'fgf_sp_5': [(0., 5.)], 'fgf_sp_10': [(0., 10.)], 'fgf_sp_60': [(0., 60.)],
                'fgf_mixed': [(1., 4.), (24., 54.), (114., 119.)]}
    failures = []
    for protocol, segments in expected.items():
        try:
            got = c.segments(protocol)
        except ValueError as error:
            failures.append(f'segments({protocol}) raised {error!r}')
            continue
        if got != segments:
            failures.append(f'segments({protocol}) = {got}, expected {segments}')
    if c.segments('fgf_sus') != [(0., 1000.)]:
        failures.append('sustained schedule changed')
    if c.segments('fgf_3_20')[:3] != [(0., 3.), (23., 26.), (46., 49.)]:
        failures.append('3/20 schedule changed')
    return failures


def check_pilot_result_fields():
    """Saved pilot results carry the validation, time-support and constant baselines."""
    result = json.loads((c.OUT / 'calibration_results.json').read_text(encoding='utf-8'))
    failures = []
    for forecast in result['forecasting']:
        horizon = forecast['horizon_min']
        for name, model in forecast['models'].items():
            if 'validation_condition_equal_mse' not in model:
                failures.append(f'{horizon:g} min {name}: no validation MSE')
            support = model['test'].get('time_support')
            if support is None:
                failures.append(f'{horizon:g} min {name}: no test time-support split')
                continue
            n = sum(part['n_transitions'] for part in (support['within'], support['beyond']) if part)
            if n != model['test']['n_transitions']:
                failures.append(f'{horizon:g} min {name}: time-support rows {n} != {model["test"]["n_transitions"]}')
    if 'training_mean_constant' not in result['input_output_identification']:
        failures.append('open-loop: no training-mean constant')
    for token, models in result['common_window_open_loop_comparison']['condition_rmse'].items():
        if 'training_mean_constant' not in models:
            failures.append(f'common window {token}: no training-mean constant')
    return failures


def check_manuscript_assets():
    """Every table and figure the manuscript includes exists, and the frozen reference covers the expanded analysis."""
    failures = []
    tex = (ROOT / 'paper/main.tex').read_text(encoding='utf-8')
    for name in re.findall(r'\\input\{tables/([^}]+)\}', tex):
        path = ROOT / 'paper/tables' / (name if name.endswith('.tex') else name + '.tex')
        if not path.is_file():
            failures.append(f'missing paper/tables/{path.name}')
    for name in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', tex):
        if not (ROOT / 'paper' / name).is_file():
            failures.append(f'missing paper/{name}')
    open_loop = ROOT / 'paper/tables/open_loop_rmse.tex'
    if open_loop.is_file() and 'constant' not in open_loop.read_text(encoding='utf-8'):
        failures.append('open-loop table lacks the training-mean constant')
    reference = json.loads((ROOT / 'reproduction/reference_metrics.json').read_text(encoding='utf-8'))['metrics']
    if not any(key.startswith('expanded/') for key in reference):
        failures.append('frozen reference has no expanded metrics')
    return failures


def check_manuscript_numbers_frozen():
    """Every number printed with four or more decimals in the manuscript must
    equal a frozen reference value at the printed precision."""
    tex = (ROOT / 'paper/main.tex').read_text(encoding='utf-8')
    reference = json.loads((ROOT / 'reproduction/reference_metrics.json').read_text(encoding='utf-8'))['metrics']
    frozen = [float(v) for v in reference.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    failures = []
    for text in re.findall(r'(?<![\d.])(-?\d+\.\d{4,})(?![\d])', tex):
        places = len(text.split('.')[1])
        if not any(round(v, places) == float(text) for v in frozen):
            failures.append(f'manuscript number {text} is not in the frozen reference')
    return failures


def check_validation_kit():
    """Planning outputs match the current draft registration and never report results."""
    failures = []
    config_raw = (KIT / 'configs/validation_registration_draft.json').read_bytes()
    config, config_sha = json.loads(config_raw), hashlib.sha256(config_raw).hexdigest()
    load = lambda name: json.loads((KIT / 'planning_outputs' / name).read_text(encoding='utf-8'))
    audit = load('registration_audit.json')
    if audit['config_sha256'] != config_sha:
        failures.append('registration audit was produced from a different config')
    if config['status'].startswith('draft') and audit['ready_assertions_complete']:
        failures.append('draft registration reported as ready')
    for stage in ('pilot', 'main'):
        record = load(f'{stage}_candidate_design.json')
        rows = (KIT / f'planning_outputs/{stage}_candidate_design.csv').read_text(encoding='utf-8').strip().split('\n')[1:]
        if record['config_sha256'] != config_sha or record['candidate_variants'] != len(rows):
            failures.append(f'{stage} candidate design is stale or inconsistent')
        if record['biological_sessions_collected'] != 0 or record['hardware_commands_generated']:
            failures.append(f'{stage} candidate design claims collected sessions or hardware commands')
    templates = sorted((KIT / 'data_templates').glob('*.csv'))
    filled = [t.name for t in templates if len(t.read_text(encoding='utf-8').strip().split('\n')) != 1]
    if filled:
        failures.append(f'data templates must stay header-only (put real records elsewhere): {filled}')
    contract = load('data_contract_audit.json')
    if not filled and (contract['status'] != 'awaiting_actual_records' or contract['empirical_task_reliability'] is not None):
        failures.append('empty templates must give awaiting_actual_records and no reliability')
    if load('software_verification.json')['status'] != 'passed':
        failures.append('planning-tool software verification did not pass')
    return failures


def check_validation_kit_provenance():
    """Received files are byte-identical, differ only in line endings, or are listed as edited."""
    received = json.loads((KIT / 'source_package_manifest.json').read_text(encoding='utf-8'))['files']
    failures = []
    for item in received:
        raw = (ROOT / item['path']).read_bytes()
        same = hashlib.sha256(raw).hexdigest() == item['sha256']
        line_endings_only = hashlib.sha256(raw.replace(b'\n', b'\r\n')).hexdigest() == item['sha256']
        if not (same or line_endings_only or item['path'] in KIT_EDITED):
            failures.append(f'received file changed without an integration record: {item["path"]}')
    return failures


CHECKS = (check_outputs_lf, check_schedules, check_pilot_result_fields, check_manuscript_assets,
          check_manuscript_numbers_frozen, check_validation_kit, check_validation_kit_provenance)


def main():
    failures = {check.__name__: check() for check in CHECKS}
    failures = {name: items for name, items in failures.items() if items}
    if failures:
        for name, items in failures.items():
            print(f'FAIL {name}: {len(items)}')
            for item in items:
                print('   ', item)
        return 1
    print(f'pipeline checks passed ({len(CHECKS)} checks)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
