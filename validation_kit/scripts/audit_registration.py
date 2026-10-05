"""Audit a prospective registration draft; assertions do not prove real calibration."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PARTITIONS=('train','model_selection','probability_calibration','locked_test')
REQUIRED=(
 'group_split.registered_largest_shared_unit','task.task_definitions',
 'task.joint_same_cell_fraction_threshold','task.eligible_cell_cohort_rule',
 'task.missing_response_and_tracking_rule','task.sequence_reliability_target',
 'task.false_activation_rule','task.deadline_and_completion_rule',
 'statistics.prediction_practical_improvement_margin','statistics.cost_practical_improvement_margin_min',
 'statistics.reliability_noninferiority_margin','statistics.confidence_and_multiplicity_procedure',
 'statistics.fixed_test_sample_size','statistics.pilot_variance_estimate_source',
 'models_and_control.model_candidates_frozen_before_new_test',
 'models_and_control.same_feedback_and_future_input_budget_verified',
 'models_and_control.mechanistic_filter_validated',
 'models_and_control.response_uncertainty_and_task_probability_calibrated',
 'models_and_control.supported_action_set','models_and_control.reference_and_fallback_protocol',
 'models_and_control.maximum_compute_plus_actuation_latency_min',
 'measurement.readout_platform_confirmed','measurement.command_and_observation_clock_synchronised',
 'measurement.feedback_arrival_logged','measurement.native_gap_and_initial_baseline_rules_confirmed',
 'measurement.delivery_timing_evidence_and_scope','measurement.constant_flow_or_validated_flow_compensation',
 'evidence.platform_and_calibration_record_paths','evidence.pilot_record_paths','evidence.raw_source_checksums_record',
 'evidence.freeze_record')


def get(config,path):
    value=config
    for key in path.split('.'):value=value.get(key) if isinstance(value,dict) else None
    return value


def audit(config):
    missing=[n for n in REQUIRED if get(config,n) is None or get(config,n) is False or get(config,n)==[] or get(config,n)=='']
    errors=[];owners={}
    for partition in PARTITIONS:
        groups=config['group_split'][partition]
        if not groups:missing.append('group_split.'+partition)
        for group in groups:
            if not isinstance(group,str) or not group.strip():errors.append('Invalid actual split-group ID')
            if group in owners:errors.append(f'Shared group {group} across or repeated within splits')
            owners[group]=partition
    cohorts=config['group_split'].get('locked_test_subcohorts',{})
    registered_test=set(config['group_split']['locked_test'])
    cohort_groups=[]
    for name in ('prediction_validation','prospective_control'):
        groups=cohorts.get(name,[])
        if not groups:missing.append('group_split.locked_test_subcohorts.'+name)
        cohort_groups.extend(groups)
    if len(set(cohort_groups))!=len(cohort_groups):errors.append('Locked prediction/control cohorts overlap')
    if set(cohort_groups)!=registered_test:errors.append('Locked test cohort union differs from registered locked test groups')
    for field in ('task.joint_same_cell_fraction_threshold','task.sequence_reliability_target'):
        value=get(config,field)
        if value is None:continue
        if isinstance(value,bool) or not isinstance(value,(int,float)):
            errors.append(field+' must be a numeric proportion');continue
        valid=0<value<=1 if field.endswith('fraction_threshold') else 0<value<1
        if not valid:errors.append(field+' has an invalid probability range')
    m=config['measurement']
    if m['tracer_is_absolute_FGF2']:errors.append('Tracer cannot be asserted to be absolute FGF2 without an independently specified conversion model')
    if m['measured_mass_claim_allowed'] and not (m['flow_measured'] is True and m['inlet_FGF2_assayed'] is True):
        errors.append('Mass claim requires measured flow and inlet FGF2')
    if m['measured_local_exposure_claim_allowed'] and m['receiver_FGF2_assayed'] is not True:
        errors.append('Local exposure claim requires measured receiver FGF2')
    tasks=config['task']['task_definitions']
    if tasks is not None:
        if not isinstance(tasks,list) or len(tasks)<2:errors.append('At least two ordered task definitions required')
        else:
            for t in tasks:
                if not isinstance(t,dict) or any(t.get(k) is None for k in ('response_lower','response_upper','hold_min','deadline_min','minimum_native_frames')):
                    errors.append('Incomplete task definition');continue
                if not t['response_lower']<t['response_upper'] or t['hold_min']<=0 or t['deadline_min']<=0 or t['minimum_native_frames']<1:
                    errors.append('Invalid task band/interval')
    ready=not missing and not errors
    return {'status':'registration_fields_complete_pending_evidence_review' if ready else 'draft_not_ready',
            'missing_required_fields':missing,'errors':errors,'ready_assertions_complete':ready,
            'physical_calibration_verified_by_this_script':False,'independent_test_completed':False,
            'scope':'Structural audit only. Inspect evidence and freeze before a prospective test; it does not prove independence or instrument accuracy.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=ROOT/'configs/validation_registration_draft.json')
    p.add_argument('--output',type=Path,default=ROOT/'planning_outputs/registration_audit.json')
    p.add_argument('--require-ready',action='store_true');a=p.parse_args();r=audit(json.loads(a.config.read_text()))
    r['config_sha256']=hashlib.sha256(a.config.read_bytes()).hexdigest();a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,indent=2)+'\n',newline='\n');print(json.dumps(r,indent=2))
    return 2 if a.require_ready and not r['ready_assertions_complete'] else 0


if __name__=='__main__':raise SystemExit(main())
