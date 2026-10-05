"""Generate draft candidate slots, never biological observations or pump commands."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random

ROOT=Path(__file__).resolve().parents[1]


def build(config,stage):
    d=config['candidate_design'];rows=[]
    clocks=d['probe_clock_blocks_min'][:1] if stage=='pilot' else d['probe_clock_blocks_min']
    waits=d['pilot_waits_min'] if stage=='pilot' else d['waits_min']
    doses=[d['pilot_probe_command_ng_per_ml']] if stage=='pilot' else d['probe_command_candidates_ng_per_ml']
    for clock in clocks:
        for duration in d['history_durations_min']:
            for wait in waits:
                for dose in doses:
                    rows.append({'condition_type':'history_then_probe','pre_duration_min':duration,'wait_min':wait,
                                 'pre_command_ng_per_ml':d['pre_command_ng_per_ml'],'probe_command_ng_per_ml':dose,
                                 'planned_probe_clock_min':clock})
                rows.append({'condition_type':'matched_pre_only','pre_duration_min':duration,'wait_min':wait,
                             'pre_command_ng_per_ml':d['pre_command_ng_per_ml'],'probe_command_ng_per_ml':0.,
                             'planned_probe_clock_min':clock})
        for dose in doses:
            rows.append({'condition_type':'probe_only','pre_duration_min':0.,'wait_min':None,
                         'pre_command_ng_per_ml':0.,'probe_command_ng_per_ml':dose,'planned_probe_clock_min':clock})
        rows.append({'condition_type':'buffer_only','pre_duration_min':0.,'wait_min':None,
                     'pre_command_ng_per_ml':0.,'probe_command_ng_per_ml':0.,'planned_probe_clock_min':clock})
    for index,r in enumerate(rows,1):
        r['planned_condition_id']=f'{stage}_C{index:03d}'
        r['planned_pre_start_min']=None if not r['pre_duration_min'] else r['planned_probe_clock_min']-r['wait_min']-r['pre_duration_min']
        r['planned_pre_end_min']=None if not r['pre_duration_min'] else r['planned_probe_clock_min']-r['wait_min']
        r['planned_probe_duration_min']=d['probe_duration_min'] if r['probe_command_ng_per_ml'] else 0.
        r['planned_record_until_min']=r['planned_probe_clock_min']+d['probe_duration_min']+d['post_probe_observation_window_min']
        r.update({'actual_day_id':'','actual_session_id':'','actual_chamber_id':'','actual_cell_batch_id':'',
                  'status':'candidate_only_not_calibrated_not_executed'})
        if r['planned_pre_start_min'] is not None and r['planned_pre_start_min']<0:
            raise ValueError('Candidate clock cannot accommodate preconditioning and waiting')
    random.Random(d['planning_permutation_seed']).shuffle(rows)
    for n,r in enumerate(rows,1):r['provisional_candidate_order']=n
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/validation_registration_draft.json')
    p.add_argument('--stage',choices=['pilot','main'],default='pilot')
    p.add_argument('--output',type=Path)
    a=p.parse_args();c=json.loads(a.config.read_text());rows=build(c,a.stage)
    out=a.output or ROOT/'planning_outputs'/f'{a.stage}_candidate_design.csv';out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    record={'stage':a.stage,'candidate_variants':len(rows),'biological_sessions_collected':0,
            'planned_matrix_not_actual_randomization':True,'hardware_commands_generated':False,
            'config_sha256':hashlib.sha256(a.config.read_bytes()).hexdigest(),
            'note':'Actual day/chamber/block assignment and calibrated concentrations must be recorded separately before use.'}
    out.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n',newline='\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
