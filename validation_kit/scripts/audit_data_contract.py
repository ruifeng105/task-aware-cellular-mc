"""Check actual group/split identity and feedback times; empty templates yield no results."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=['session_id','day_id','chamber_id','split_group_id','cell_batch_id','device_batch_id','partition','test_cohort',
          'planned_condition_id','eligible_cell_count_before_command','raw_data_path','raw_data_sha256']
OBS=['session_id','sequence_id','cell_id','time_min','arrival_time_min','normalized_fret','raw_fret_ratio',
     'response_missing','missing_reason','eligible_at_sequence_start']
PARTITIONS={'train','model_selection','probability_calibration','locked_test','pilot'}


def validate(manifest,observations):
    if not set(MANIFEST)<=set(manifest.columns) or not set(OBS)<=set(observations.columns):
        raise ValueError('Required manifest/observation columns missing')
    if manifest.empty and observations.empty:
        return {'status':'awaiting_actual_records','actual_sessions':0,'actual_observations':0,
                'empirical_task_reliability':None,'measured_mass_ng':None,'measured_local_exposure':None}
    if manifest.empty or observations.empty:raise ValueError('Manifest and observations must both be supplied')
    for col in ['session_id','day_id','chamber_id','split_group_id','cell_batch_id','device_batch_id','partition']:
        if manifest[col].isna().any() or manifest[col].astype(str).str.strip().eq('').any():raise ValueError('Missing actual '+col)
    if manifest.session_id.duplicated().any():raise ValueError('Duplicate actual session ID')
    if not set(manifest.partition)<=PARTITIONS:raise ValueError('Unknown actual partition')
    for col in ['day_id','split_group_id']:
        if (manifest.groupby(col).partition.nunique()>1).any():raise ValueError('Shared '+col+' crosses data splits')
    test=manifest.loc[manifest.partition.eq('locked_test')]
    if not set(test.test_cohort)<= {'prediction_validation','prospective_control'}:raise ValueError('Locked test role missing/invalid')
    for col in ['day_id','split_group_id']:
        if (test.groupby(col).test_cohort.nunique()>1).any():raise ValueError('Locked prediction/control cohorts share '+col)
    if set(observations.session_id)!=set(manifest.session_id):raise ValueError('Manifest/observation sessions differ')
    for c in ('sequence_id','cell_id'):
        if observations[c].isna().any() or observations[c].astype(str).str.strip().eq('').any():raise ValueError('Missing actual '+c)
    obs=observations.copy()
    for c in ('time_min','arrival_time_min','normalized_fret','raw_fret_ratio'):obs[c]=pd.to_numeric(obs[c],errors='raise')
    if not np.isfinite(obs[['time_min','arrival_time_min']]).all().all():raise ValueError('Missing/nonfinite clock')
    if (obs.arrival_time_min<obs.time_min).any():raise ValueError('Feedback arrival before acquisition')
    if obs.duplicated(['session_id','cell_id','time_min']).any():raise ValueError('Duplicate cell observation time')
    for c in ('response_missing','eligible_at_sequence_start'):
        if not obs[c].astype(str).str.lower().isin(['0','1','false','true']).all():raise ValueError('Invalid binary field '+c)
    missing=obs.response_missing.astype(str).str.lower().isin(['1','true'])
    if obs.loc[missing,'normalized_fret'].notna().any() or obs.loc[~missing,'normalized_fret'].isna().any():
        raise ValueError('Missing response flag/value mismatch; do not fill losses with zero')
    if np.isinf(obs.normalized_fret.dropna()).any():raise ValueError('Infinite response')
    if obs.loc[missing,'missing_reason'].isna().any() or obs.loc[missing,'missing_reason'].astype(str).str.strip().eq('').any():
        raise ValueError('Missing observation requires an actual missingness reason')
    if (obs.groupby(['session_id','sequence_id','cell_id']).eligible_at_sequence_start.nunique()>1).any():
        raise ValueError('Baseline eligible cohort changed within a sequence')
    return {'status':'structural_checks_passed','actual_sessions':len(manifest),'actual_observations':len(obs),
            'distinct_day_ids':manifest.day_id.nunique(),'declared_shared_groups':manifest.split_group_id.nunique(),
            'empirical_task_reliability':None,'measured_mass_ng':None,'measured_local_exposure':None,
            'scope':'No scoring, raw-source checksum verification, task classification, or experimental-independence proof is performed here.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=ROOT/'data_templates/session_manifest.csv')
    p.add_argument('--observations',type=Path,default=ROOT/'data_templates/observations.csv')
    p.add_argument('--output',type=Path,default=ROOT/'planning_outputs/data_contract_audit.json')
    a=p.parse_args();r=validate(pd.read_csv(a.manifest,dtype=str),pd.read_csv(a.observations,dtype=str))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n',newline='\n');print(json.dumps(r,indent=2))


if __name__=='__main__':main()
