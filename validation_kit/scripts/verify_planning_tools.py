"""Software/analytic checks only; fixtures are not newly collected experiments."""
from pathlib import Path
import json
import math
import pandas as pd
from scipy.stats import binomtest
from design_candidates import build
from sample_size import lower_bound,planning
from audit_data_contract import validate,MANIFEST,OBS
from audit_registration import audit

ROOT=Path(__file__).resolve().parents[1]


def main():
    c=json.loads((ROOT/'configs/validation_registration_draft.json').read_text())
    pilot=build(c,'pilot');main_rows=build(c,'main')
    assert len(pilot)==10 and len(main_rows)==54
    assert all(r['actual_day_id']==r['actual_session_id']==r['actual_chamber_id']=='' for r in main_rows)
    assert len({r['planned_condition_id'] for r in main_rows})==54
    for r in main_rows:
        if r['pre_duration_min']:
            assert r['planned_pre_end_min']-r['planned_pre_start_min']==r['pre_duration_min']
            assert r['planned_probe_clock_min']-r['planned_pre_end_min']==r['wait_min']
        if r['condition_type']=='history_then_probe':
            assert any(z['condition_type']=='matched_pre_only' and all(z[k]==r[k] for k in
                ('pre_duration_min','wait_min','pre_command_ng_per_ml','planned_probe_clock_min')) for z in main_rows)
    errors=[]
    for k,n in ((0,10),(7,10),(29,29),(59,59),(299,299)):
        reference=binomtest(k,n,alternative='greater').proportion_ci(confidence_level=.95,method='exact').low
        errors.append(abs(lower_bound(k,n)-reference))
    assert max(errors)<1e-10
    for r in planning()[0]:
        n=r['n_independent_all_success_trials'];q=r['target']
        assert lower_bound(n,n)>=q and lower_bound(n-1,n-1)<q
        assert abs(lower_bound(n,n)-math.pow(.05,1/n))<1e-12
    empty=validate(pd.DataFrame(columns=MANIFEST),pd.DataFrame(columns=OBS));assert empty['empirical_task_reliability'] is None
    assert not audit(c)['ready_assertions_complete']
    # Explicit synthetic in-memory records test timing and split leakage rejection.
    m=pd.DataFrame([{k:'SW_fixture' for k in MANIFEST}]);m['partition']='pilot'
    o=pd.DataFrame([{k:'SW_fixture' for k in OBS}]);o=o.assign(time_min=0.,arrival_time_min=.1,
        normalized_fret=1.,raw_fret_ratio=1.,response_missing=0,eligible_at_sequence_start=1)
    assert validate(m,o)['status']=='structural_checks_passed'
    delayed=o.copy();delayed['arrival_time_min']=-.1
    try:validate(m,delayed)
    except ValueError:pass
    else:raise AssertionError('Invalid availability clock accepted')
    leak=pd.concat([m,m.assign(session_id='SW_fixture_second',partition='locked_test',test_cohort='prediction_validation')],ignore_index=True)
    obs=pd.concat([o,o.assign(session_id='SW_fixture_second')],ignore_index=True)
    try:validate(leak,obs)
    except ValueError:pass
    else:raise AssertionError('Shared experimental day/group crossed data splits')
    test_manifest=pd.concat([m.assign(partition='locked_test',test_cohort='prediction_validation'),
        m.assign(session_id='SW_fixture_second',partition='locked_test',test_cohort='prospective_control')],ignore_index=True)
    try:validate(test_manifest,obs)
    except ValueError:pass
    else:raise AssertionError('Locked prediction/control cohorts shared an experimental unit')
    r={'status':'passed','scope':'Planning software only; no biological or physical calibration evidence',
       'pilot_conditions':10,'main_candidate_variants':54,'actual_experimental_records_created':0,
       'max_exact_bound_error_against_scipy':max(errors),'planned_probe_clock_alignment_checked':True,
       'matched_no_second_command_controls_checked':True,'invalid_feedback_time_rejected':True,
       'group_split_leakage_rejected':True,'locked_test_cohort_overlap_rejected':True,'empty_data_never_produces_reliability':True,
       'draft_not_misrepresented_as_ready':True}
    out=ROOT/'planning_outputs/software_verification.json';out.write_text(json.dumps(r,indent=2)+'\n',newline='\n');print(json.dumps(r,indent=2))


if __name__=='__main__':main()
