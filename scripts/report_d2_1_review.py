"""Persist actual sampled visual review, its uncertainty and D2.1 audit evidence."""
import csv
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
out=root/'experiments/position/d2_1_evidence_completion/index1/result'
raw=root/'results/experiments/D2_1_ACTION_CHAIN_EVIDENCE_COMPLETION'
manifest=json.loads((raw/'visual_review/manifest.json').read_text())
release={0:(420,450),2:(360,390),3:(390,420),4:(390,420),5:(360,390),6:(360,390),
7:(390,420),9:(390,420),10:(390,420),12:(390,420),13:(420,450),14:(420,450),15:(420,450),
17:(420,450),18:(420,450),19:(420,450),21:(420,450),22:(390,420),25:(420,450),28:(420,450),
30:(330,360),32:(420,450),33:(420,450),34:(420,450),35:(420,450),36:(420,450),39:(420,450),
40:(420,450),42:(390,420),43:(390,420),45:(420,450),49:(420,450),50:(420,450),51:(420,450),53:(420,450)}

def save(name,data):
    (out/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

with (out/'demonstration_events.csv').open('w',encoding='utf-8',newline='') as f:
    w=csv.writer(f)
    w.writerow(['split','episode','event','status','frame_lower','frame_upper','time_lower_s','time_upper_s','source','visibility','evidence','reason'])
    for m in manifest:
        ep=m['episode'];m['review_status']='SAMPLED_DUAL_VIEW_REVIEWED_BY_CODEX'
        m['full_frame_review_completed']=False
        m['review_note']='Whole temporal range sampled every 1s; short boundaries not resolved; top occlusion and moving wrist perspective.'
        m['dense_review_needed']=[{'frames':[150,240],'events':['effective_closure','lift_off']},
                                  {'frames':[210,390],'events':['lower_start','top_hold']},
                                  {'frames':[330,478],'events':['table_contact','release']}]
        for event in ['effective_closure','lift_off','lower_start','table_contact','release','top_hold']:
            bounds=release.get(ep) if event=='release' else None
            status='VISIBLE_RELEASE_CANDIDATE_INTERVAL' if bounds else 'UNKNOWN'
            reason={'effective_closure':'Jaws and cup move together later, but contact hidden; effective clamp instant unconfirmed.',
                'lift_off':'Cup bottom/table gap occluded in top; wrist viewpoint moves with gripper.',
                'lower_start':'Continuous posture change visible; exact vertical descent onset unresolved at 1s sampling.',
                'table_contact':'Cup bottom occluded; release alone cannot timestamp first table contact.',
                'release':'Cup separation/open jaws observed between sampled frames; late release unresolved if no interval.',
                'top_hold':'Sampling cannot distinguish short hold from continuous reversal; neither HOLD nor NO_HOLD asserted.'}[event]
            w.writerow([m['split'],ep,event,status,bounds[0] if bounds else '',bounds[1] if bounds else '',
                        bounds[0]/30 if bounds else '',bounds[1]/30 if bounds else '',
                        'VIDEO_REVIEWED_BY_CODEX','partial',m['sheet'],reason])
save('visual_review_manifest.json',manifest)
save('demonstration_audit.json',{'train_reviewed_sampled':36,'dev_reviewed_sampled':12,
    'full_frame_review_episodes':0,'release_candidate_intervals':len(release),
    'verified_lift_hold_intervals':0,'event_timing_resolution_s':1,
    'hold_duration_s':None,'lift_height_mm':None,'automatic_force_labels_used':False,
    'review_limitations':'Sampled review is completed; full-video dense event review still incomplete.'})
with (out/'per_episode_phase_metrics.csv').open('w',encoding='utf-8',newline='') as f:
    w=csv.writer(f);w.writerow(['model','episode','phase','frames','arm_mae','gripper_mae','status'])
    for model in ['P1','R3']:
        data=json.loads((root/f'experiments/position/d2_action_chain_diagnosis/index1/result/{model.lower()}_unphased_metrics.json').read_text())
        for ep,x in data['episodes'].items():
            w.writerow([model,ep,'unknown',x['frames'],sum(x['h0_per_joint_mae'][:5])/5,x['h0_per_joint_mae'][5],'UNPHASED'])
            for phase in ['approach','grasp','lift','airborne_hold','lower','release']:
                w.writerow([model,ep,phase,'','','','BLOCKED_RELIABLE_EVENT_BOUNDARIES'])
save('prediction_diagnosis.json',{'diagnostic_category':'UNRESOLVED','phase_metrics_valid':False,
    'why':'Sampled visuals do not resolve lift/hold/contact boundaries; no proxy labels substituted.',
    'P1_dev_episodes':12,'R3_dev_episodes':12,'teacher_forced':True,'closed_loop':'UNVERIFIED'})
save('postprocess_replay.json',{'source':'D2/result/postprocess_replay.json',
    'normalized_equivalence':'PASS see replay_equivalence_validation.json',
    'actual_temporal_and_clamp_validation':'D2/result/replay_validation.json PASS',
    'four_configs':['off+5','off+2','0.01+5','0.01+2'],'phase_effect':None,
    'actual_runtime':'UNVERIFIED','fresh_motor_state_vs_recorded_state':'Not equivalent robot conditions'})
save('deployment_contract_audit.json',{'n_action_steps_off':1,'ensemble_on':'full chunk every step',
    'queue_difference':'No multiple-step queued execution with n_action_steps=1',
    'episode_reset':True,'runtime_parameters':'UNVERIFIED_NO_RUNTIME_LOGS',
    'D2_static_audit_stale_blocker':'CHECKPOINTS_AND_TORCH resolved on server; local torch still absent'})
save('report_consistency_audit.json',{'original_D2_preserved':True,
    'D2_initial_cache_full_chunk_ground_truth_evaluated':False,
    'D2_later_replay_validation_full_chunk_reconstructed':True,
    'current_horizon_metrics':'Computed separately on same-episode targets and masks',
    'old_hardware_and_torch_blockers':'historical; server inference occurred',
    'new_visual_status':'sampled reviewed; full event review incomplete',
    'pure_ACT_claim':'RETRACTED_B3_IS_OCACT'})
save('metrics.json',{'train_sampled_reviews':36,'dev_sampled_reviews':12,'release_candidate_intervals':len(release),
    'full_frame_review_episodes':0,'reliable_lift_hold_coverage':0,'success_rate':None,
    'training':False,'final_test':False,'real_robot':False})
save('validation.json',{'DIAGNOSIS_COMPLETE':False,'acceptance_status':'PARTIAL_PASS',
    'diagnostic_category':'UNRESOLVED','B3_identity_status':'OCACT_VERIFIED',
    'training':False,'final_test':False,'real_robot':False,
    'G0':'IDENTITY_AND_ASSETS_VERIFIED_CACHE_PROTOCOL_PARTIAL',
    'G1':'PARTIAL_VISUAL_BOUNDARIES','G2':'NORMALIZED_EQUIVALENCE_PASS',
    'G3':'NA','G4':'B3_FORWARD_PATH_VERIFIED','G5':'PARTIAL_NO_RELIABLE_PHASE_METRICS',
    'G6':'OFFLINE_ONLY_RUNTIME_UNVERIFIED','G7':'NA','G8':'UNRESOLVED'})
save('evidence_manifest.json',{'raw':str(raw),'scripts':['prepare_d2_1_visual_review.py','audit_d2_1_remote.py','report_d2_1_review.py'],
    'result_sha256':{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in out.rglob('*') if p.is_file() and p.name!='evidence_manifest.json'},
    'missing':['dense visual event review','reliable stage metrics','runtime logs'],
    'Final_episode_contents_read':False})
print('48 dual-view sampled reviews recorded;',len(release),'release candidate intervals; stage metrics still blocked')
