"""Follow the R4 compute queue and package factual executor reports, never train."""
import csv
import json
import shutil
import time
import zipfile
from pathlib import Path

from run_r4_repro import WORK, OUT, F5, save, table, digest


def finalize():
    state=json.loads((OUT/'queue_state.json').read_text())
    assert state['status']=='COMPUTE_DONE_REQUIRES_REVIEW_AND_RESULT'
    gate=json.loads((OUT/'repro_gate.json').read_text())
    assert gate['R4_REPRODUCIBILITY_PASS']
    runtime={}
    for mode,seeds in [('historical',[3]),('benchmark_off',[3,1])]:
        for seed in seeds:
            directory=WORK/'raw'/mode/f'S{seed}'
            for name in ['audit_config.json','B0/audit_config.json','B1/audit_config.json']:
                runtime[f'{mode}/S{seed}/{name}']=json.loads((directory/name).read_text())
    save(OUT/'audit_config.json',dict(runtime=runtime,decision_sha256=digest(WORK/'decision.md'),
        scripts={p.name:digest(p) for p in WORK.glob('*.py')},
        checkpoint_files={str(p):digest(p) for seed in [1,2,3,4]
            for p in (F5/'runs'/f'G60-S{seed}'/'checkpoints/080000/pretrained_model').iterdir()
            if p.is_file() and p.name in ['config.json','model.safetensors','f2_1_normalization.json']},
        original_f5_source_snapshot_sha256=digest(F5/'source_snapshot.json'),
        normalization_sha256=digest(F5/'inputs/normalization.json'),
        prototype_sha256=digest(Path('/root/autodl-tmp/F2_1_r1_20261019/inputs/cup_prototype_bank_v1.pt'))))
    for filename in ['same_process_repeat.csv','cross_process_repeat.csv','preprocess_repeat.csv']:
        records=[]
        for mode,seeds in [('historical',[3]),('benchmark_off',[3,1])]:
            for seed in seeds:
                path=WORK/'raw'/mode/f'S{seed}'/filename
                if path.exists():
                    with path.open() as stream:records.extend(csv.DictReader(stream))
        assert records
        table(OUT/filename,records)
    interpretation=json.loads((OUT/'postfix_f5_1/interpretation.json').read_text())
    save(OUT/'metrics.json',dict(R4_COMPLETE=False,R4_EXECUTION_COMPLETE=True,R4_REPRODUCIBILITY_PASS=True,
        outcome='REPRODUCIBILITY_FIXED_RUNTIME',max_difference=max(r['max_native'] for r in gate['results']),
        action_limit_exclusive=.001,postfix_interpretation=interpretation,
        external_builder_review='PENDING',external_red_team_review='PENDING',
        final_dataset_collection='NOT_EXECUTED; next approved decision required',robot_trials=0,new_training_runs=0))
    (OUT/'summary.md').write_text('# R4 execution result\n\n'
        'Outcome: REPRODUCIBILITY_FIXED_RUNTIME. R4_REPRODUCIBILITY_PASS=true.\n'
        'R4_EXECUTION_COMPLETE=true; R4_COMPLETE=false pending independent external Builder/Red-Team acceptance.\n\n'
        'Same-process frozen-input outputs were identical. Historical cross-process outputs diverged first at the observed backbone hook. '
        'Only cudnn.benchmark changed True to False; matmul/cudnn TF32 stayed enabled. Fixed A/B/C and full D on seed3/seed1 passed the original <0.001 gate. '
        'Specific cuDNN kernel IDs were not captured; do not claim operator-level localization.\n\n'
        'All critical globals were freshly replayed, frozen locals recomposed on fresh globals, and the original grouping/matching/classification code reused. '
        'See postfix_f5_1_metrics.csv and postfix_f5_1/ for numerical results and conflicts, not pre-fix numbers. '
        'All Full60 findings remain TRAIN-SET / COMMON-DIAGNOSTIC ONLY, NOT HELD-OUT GENERALIZATION. '
        'No training, data collection, real robot, actual deployment configuration changes or publication.\n',encoding='utf-8')
    (OUT/'validation.md').write_text('# Executor validation\n\n'
        'A/B/C/D prerequisite gates enforced by scripts; original tolerance retained; original F5 files untouched. '
        'Full D checks all 100 horizons / 6 actions and every input-batch hash, IDs/state/load/targets, plus h0 aggregate differences. '
        'Final D summaries for both seeds are in full_replay_repeat.csv. Fixtures and complete predictions remain under server raw/. '
        'No independent external review or real-robot equivalence is claimed.\n',encoding='utf-8')
    (OUT/'temporal_state_audit.md').write_text('# Temporal/cache audit\n\n'
        'Frozen checkpoint temporal_ensemble_coeff=null. Evaluator calls predict_action_chunk directly, not select_action; '
        'no temporal ensemble, action-queue consumption, or observation history is used. eval disables VAE sampling and dropout. '
        'Probe logs queue before/after; full replay resets every batch and logs empty action queue. '
        'Per-batch episode/frame manifests preserve episode boundaries; stateless evaluation does not require sequential carryover. '
        'This is NOT an audit of the actual robot temporal-ensemble controller.\n',encoding='utf-8')
    (OUT/'determinism_config.md').write_text('# Runtime control\n\n'
        'Historical: cudnn.benchmark=True, cudnn.deterministic=False, deterministic_algorithms=False, matmul/cudnn TF32=True. '
        'Final audit: benchmark=False only; TF32 remains True; no new training seeds or precision/model changes. '
        'Original RNG hashes, library versions, dropout/eval flags and cross-process configs are in audit_config.json. '
        'Worker seeds use the DataLoader default torch-seed+worker-id policy. No random replay augmentation was configured. '
        'Passed on this server/version/checkpoint set; portability to another GPU/runtime is not established.\n',encoding='utf-8')
    (OUT/'builder_review.md').write_text('# Executor Builder checklist (not independent review)\n\n'
        'Audit hierarchy and second-seed verification completed; full replay gate passed; critical metrics recomputed. '
        'Scientific thresholds, weights, data and training unchanged. External Builder Review PENDING.\n',encoding='utf-8')
    (OUT/'red_team_review.md').write_text('# Executor adversarial checklist (not independent review)\n\n'
        'Local frozen fixtures do not establish full-dataset repeatability without D; D has now completed for both seeds. '
        'Benchmark-off changes evaluation numerics and therefore old summaries were not reused. '
        'Runtime cause supported by controlled B comparison; exact kernel-ID mechanism unobserved. '
        'No train/generalization/robot claim upgrade, no deletion of failed historical comparisons, no seed/cherry-pick rescue. '
        'External Red-Team Review PENDING.\n',encoding='utf-8')
    (OUT/'proposal.md').write_text('# Next decision proposal, not automatic execution\n\n'
        'First externally review R4 and the postfix metric conflicts. If accepted, prepare a separate FINAL_DATASET_V1 collection/QC/split-freeze contract '
        'for 120-150 demonstrations, continuous position jitter and consistent 0.8-1.0s hold. '
        'Do not start collection, scaling, SWA/SAM, training or robot execution in R4.\n',encoding='utf-8')
    shutil.copy2(WORK/'decision.md',OUT/'decision.md')
    save(OUT/'manifest.json',dict(sha256={str(p.relative_to(OUT)):digest(p) for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json'}))
    archive=WORK/'R4_result.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as bundle:
        for p in OUT.rglob('*'):
            if p.is_file():bundle.write(p,'result/'+p.relative_to(OUT).as_posix())
    with zipfile.ZipFile(archive) as bundle:assert bundle.testzip() is None
    save(WORK/'delivery_receipt.json',dict(path=str(archive),bytes=archive.stat().st_size,sha256=digest(archive),crc='PASS'))
    save(WORK/'delivery_state.json',dict(status='EXECUTOR_PACKAGE_READY_AWAIT_EXTERNAL_REVIEW',R4_COMPLETE=False,
        R4_REPRODUCIBILITY_PASS=True,updated_unix=time.time()))


def main():
    deadline=time.monotonic()+3*3600
    while time.monotonic()<deadline:
        path=OUT/'queue_state.json'
        if path.exists():
            state=json.loads(path.read_text())
            if state['status']=='COMPUTE_DONE_REQUIRES_REVIEW_AND_RESULT':finalize();return
            if state['status']=='STOPPED_ERROR_RETAINED':
                save(WORK/'delivery_state.json',dict(status='STOPPED_COMPUTE_ERROR',R4_COMPLETE=False));return
        time.sleep(15)
    raise TimeoutError('Compute queue did not finish within follower window; evidence retained')


if __name__=='__main__':main()
