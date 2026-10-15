"""D2 allowed-split audit and conservative report; never infer visual events from force."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/position/d2_action_chain_diagnosis/index1/result'


def save(name, data):
    (OUT/name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    split = json.loads((ROOT/'experiments/data/d1_training_readiness/index1/result/split_manifest_v1.json').read_text())
    allowed = [e for e in split['episodes'] if e['split'] in ('train', 'dev')]
    ids = [e['episode_index'] for e in allowed]
    # Filter the shared parquet to permitted episodes; never expose Final rows.
    frame = pd.read_parquet(ROOT/'数据集/formal4/kind_merged_nomaster/data/chunk-000/file-000.parquet',
                            filters=[('episode_index', 'in', ids)])
    assert set(frame.episode_index) == set(ids)
    stats = []
    with (OUT/'demonstration_events.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['split', 'episode', 'event', 'frame', 'time_s', 'source', 'uncertainty', 'reason'])
        for e in allowed:
            ep = e['episode_index']
            z = frame[frame.episode_index == ep].sort_values('frame_index')
            state = np.stack(z['observation.state'])
            target = np.stack(z.action)
            timestamp = z.timestamp.to_numpy()
            stats.append({'split': e['split'], 'episode': ep, 'frames': len(z),
                'contiguous_frames': bool(np.all(np.diff(z.frame_index)==1)),
                'timestamp_step_max_error': float(np.max(np.abs(np.diff(timestamp)-1/30))),
                'finite': bool(np.isfinite(state).all() and np.isfinite(target).all()),
                'action_range': np.ptp(target,axis=0).tolist(),
                'state_range': np.ptp(state,axis=0).tolist(),
                'action_feedback_abs_difference_mean': np.abs(target-state[:,:6]).mean(0).tolist()})
            for event in ['effective_closure','lift_off','lower_start','table_contact','release']:
                w.writerow([e['split'],ep,event,'','','UNKNOWN','unknown',
                            'No Formal4 human-reviewed visual labels; visual review not completed'])
    save('demonstration_audit.json', {'episode_signal_audit':stats,
        'phase_duration_s':None,'visual_event_coverage':0,'visual_events_unknown':len(allowed)*5,
        'phase_label_source':'UNKNOWN','force_is_not_visual_ground_truth':True,
        'formal3_labels_used':False,'train_dev_selected_only':True,
        'limitation':'Signal audit is not the required full visual event audit.'})
    d = json.loads((OUT/'p1_unphased_metrics.json').read_text())
    episodes = d['episodes']
    aggregates = {}
    with (OUT/'per_episode_phase_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['model','episode','phase','frames','arm_mae','gripper_mae'])
        for ep,x in episodes.items():
            w.writerow(['P1',ep,'unknown',x['frames'],np.mean(x['h0_per_joint_mae'][:5]),x['h0_per_joint_mae'][5]])
        r3_path=OUT/'r3_unphased_metrics.json'
        if r3_path.exists():
            r3=json.loads(r3_path.read_text())
            for ep,x in r3['episodes'].items():
                w.writerow(['R3',ep,'unknown',x['frames'],np.mean(x['h0_per_joint_mae'][:5]),x['h0_per_joint_mae'][5]])
        for model in (['B3'] if r3_path.exists() else ['B3','R3']):
            for ep in episodes:
                w.writerow([model,ep,'NOT_EVALUATED','','',''])
    for key in next(iter(episodes.values()))['replay']:
        vals=[x['replay'][key] for x in episodes.values()]
        aggregates[key]={k:np.mean([x[k] for x in vals],axis=0).tolist()
                         for k in ['per_joint_mae','per_joint_clip_rate','per_joint_processing_delta']}
    paired={}
    for limit in [5,2]:
        off=aggregates[f'None_{limit}']['per_joint_mae']
        on=aggregates[f'0.01_{limit}']['per_joint_mae']
        paired[f'ensemble_effect_limit_{limit}']=(np.array(on)-off).tolist()
    save('postprocess_replay.json',{'P1_episode_equal_weight':aggregates,'paired_mae_change':paired,
        'teacher_forced':True,'actual_runtime':'UNVERIFIED','phase_specific_effect':None,
        'physical_vs_normalized_ensemble_equivalence':'NOT_TESTED',
        'clamp_equivalence_to_actual_helper':'PASS see replay_validation.json',
        'temporal_weight_equivalence':'PASS see replay_validation.json',
        'warning':'Normalization commutation not independently tested; phases unknown.'})
    save('prediction_diagnosis.json',{'category':'UNRESOLVED',
        'P1_dev_episodes':list(episodes),'frames':sum(x['frames'] for x in episodes.values()),
        'normalized_phase_mae':None,'physical_phase_mae':None,
        'full_chunk_targets_reconstructed':True,'phase_unknown_fraction':1,
        'postprocess_sensitivity_observed':True,'closed_loop':'UNRESOLVED',
        'B3_identity_conflict':'D2 says plain ACT; T3 manifest says frozen OC-ACT object branch',
        'R3_evaluation':'COMPLETE_UNPHASED' if r3_path.exists() else 'NOT_EXECUTED'})
    figs=OUT/'figures';figs.mkdir(exist_ok=True)
    keys=list(aggregates);values=[np.mean(aggregates[k]['per_joint_mae'][:5]) for k in keys]
    plt.bar(keys,values);plt.ylabel('Arm MAE (dataset units)');plt.title('P1 teacher-forced replay; phases unknown')
    plt.tight_layout();plt.savefig(figs/'p1_replay_mae.png');plt.close()
    save('metrics.json',{'P1_dev_episodes':12,'P1_frames':5748,
        'demonstration_signal_episodes':len(stats),'visual_events_unknown':len(stats)*5,
        'stage_coverage':0,'B3_coverage':None,'R3_dev_episodes':12 if r3_path.exists() else None,
        'gpu_seconds_successful_P1':d['gpu_seconds'],
        'gpu_seconds_successful_R3':r3['gpu_seconds'] if r3_path.exists() else None,
        'gpu_seconds_failed_attempt':None,'training':False,'final_test':False,'real_robot':False})
    save('validation.json',{'DIAGNOSIS_COMPLETE':False,'acceptance_status':'PARTIAL_PASS',
        'diagnostic_category':'UNRESOLVED','training':False,'final_test':False,'real_robot':False,
        'gates':{'G0':'PARTIAL_B3_IDENTITY_CONFLICT','G1':'PARTIAL_UNITS_AND_DELAY_UNVERIFIED',
        'G2':'PARTIAL_REPLAY_EQUIVALENCE_UNTESTED','G3':'NA','G4':'STATIC_ONLY',
        'G5':'PARTIAL_P1_FULL_DEV_PHASES_UNKNOWN','G6':'PARTIAL_RUNTIME_UNVERIFIED',
        'G7':'NA','G8':'UNRESOLVED'}})
    save('evidence_manifest.json', {'remote_raw':'/root/autodl-tmp/lerobot/results/experiments/D2_P1_ACTION_CHAIN_DIAGNOSIS',
        'commands':['scripts/run_d2_p1.sh','scripts/run_d2_r3.sh','scripts/verify_d2_replay.py',
                    'python scripts/report_d2_partial.py'],
        'failed_attempt':'remote p1_failed_open_files.log; descriptor limit1024 fixed65535',
        'missing':['B3 valid contract identity','48 episode visual event review',
                   'runtime logs','prototype asset hash','normalized ensemble equivalence'],
        'result_sha256':{str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in OUT.rglob('*') if p.is_file() and p.name!='evidence_manifest.json'}})
    print('Allowed signal episodes',len(stats),'P1 episodes',len(episodes))


if __name__=='__main__':
    main()
