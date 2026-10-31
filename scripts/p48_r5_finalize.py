"""Validate all authorized r5 compute, produce a compact auditable result package, then stop."""
import csv
import gc
import json
import os
import shutil
import subprocess
import time
import zipfile

os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_DATASETS_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
from p48_r5_common import OUT,ROOT,WORK,CachedData,sha,write

KINDS=['act','legacy','fixed']


def read_csv(path):
    with path.open() as stream:return list(csv.DictReader(stream))


def csv_out(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    import numpy as np
    import torch
    from p48_r5_evaluate import load
    status=json.loads((OUT/'status.json').read_text());assert status['status']=='PILOT_COMPUTE_COMPLETE'
    dev=[];rollouts=[];checkpoints=[];training=[];objects=[];latency=[];rows=[]
    data=CachedData('train')
    for kind in KINDS:
        run=OUT/'runs'/kind
        complete=json.loads((run/'training_complete.json').read_text());assert complete['updates']==10000
        d=json.loads((OUT/'evaluation'/kind/'dev_complete.json').read_text());assert d['episodes']==9 and d['samples']==1391
        r=json.loads((OUT/'rollouts'/kind/'rollout_complete.json').read_text())
        assert r['episodes']==10 and r['init_ids']==list(range(10,20)) and r['episode_seeds']==[1]*10
        dev.extend(read_csv(OUT/'evaluation'/kind/'dev9_metrics.csv'))
        rollouts.extend(read_csv(OUT/'rollouts'/kind/'libero_rollouts.csv'))
        for item in read_csv(run/'checkpoint_manifest.csv'):
            assert sha(item['path'])==item['sha256'];checkpoints.append(dict(**item,purpose='FORMAL_PILOT'))
        for item in read_csv(run/'training_metrics.csv'):training.append(dict(kind=kind,**item))
        if kind!='act':objects.extend(read_csv(OUT/'evaluation'/kind/'object_path_diagnostics.csv'))
        cfg,policy,latest=load(kind);batch=data.batch([0]);times=[]
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for i in range(60):
                torch.cuda.synchronize();started=time.perf_counter();policy.predict_action_chunk(batch);torch.cuda.synchronize()
                if i>=10:times.append(time.perf_counter()-started)
        latency.append(dict(kind=kind,batch_size=1,chunk_size=100,repeats=50,
             mean_seconds=float(np.mean(times)),median_seconds=float(np.median(times)),p95_seconds=float(np.quantile(times,.95)),
             params=sum(p.numel() for p in policy.parameters()),trainable_params=sum(p.numel() for p in policy.parameters() if p.requires_grad),
             peak_gpu_bytes=torch.cuda.max_memory_allocated(),scope='isolated after all training/rollout jobs; preprocessed input; full chunk prediction'))
        rows.append(dict(kind=kind,training_updates=10000,successes=r['successes'],rollouts=10,
                         **d['episode_equal_metrics'],training_wall_seconds=complete['elapsed_s']))
        del policy,batch;gc.collect();torch.cuda.empty_cache()
    assert len(dev)==27 and len(rollouts)==30 and len(objects)==54
    csv_out(OUT/'training_metrics.csv',training);csv_out(OUT/'dev9_metrics.csv',dev)
    csv_out(OUT/'libero_rollouts.csv',rollouts);csv_out(OUT/'checkpoint_manifest.csv',checkpoints)
    csv_out(OUT/'object_path_diagnostics.csv',objects);csv_out(OUT/'latency_and_parameters.csv',latency)
    write(OUT/'performance_summary.json',dict(pilot_only=True,rows=rows,latency=latency,
        evidence_level='single task seed1, 10k updates and10 paired simulator episodes; no real robot and no leaderboard claim'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=OUT/'figures';figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for kind in KINDS:
        selected=[x for x in training if x['kind']==kind]
        x=np.array([int(v['step']) for v in selected]);y=np.array([float(v['loss']) for v in selected])
        axes[0].plot(x,y,label=kind,alpha=.75)
    axes[0].set(xlabel='Effective optimizer updates',ylabel='Native ACT total loss',title='r5 single-task pilot');axes[0].legend()
    axes[1].bar(KINDS,[x['successes'] for x in rows]);axes[1].set(ylim=(0,10),ylabel='Successes / 10 paired initial states',title='Pilot only; not a leaderboard result')
    fig.savefig(figures/'training_and_paired_pilot.png',dpi=180);plt.close(fig)
    lines=['# P4.8-r5 complete single-task pilot','',
        '**G2_PROTOTYPE_PASS / G3_TRAINING_PASS / G4_CLOSED_LOOP_PASS**. G4 means all scheduled rollouts executed; it does not require a nonzero success count.','',
        '| Model | Updates | Simulator successes | Dev9 h0 arm raw MAE | Dev9 h0 gripper raw MAE |',
        '|---|---:|---:|---:|---:|']
    for row in rows:lines.append(f'| {row["kind"]} | 10000 | {row["successes"]}/10 | {row["h0_arm_raw_mae"]:.6f} | {row["h0_gripper_raw_mae"]:.6f} |')
    lines+=['','Raw action errors are OSC controller-input units, not SO101 joint degrees or measured end-effector motion. Dev9 is teacher-forced internal diagnosis, separate from simulator success.',
        '', 'All three jobs use the fixed Train35/Dev9 split and one seed. P1 Legacy/Fixed share initial weights, prototype bytes, data order and budget; only online DINO normalization differs. ACT has the native architecture/loss/optimizer defaults with disclosed interface,10k/seed1/sampler/cache/numerical-mode overrides.',
        '', 'The prototype method matches formal6: four fixed Train frames, four Codex-selected target-interior points, scaled35px ellipse/patch-center sampling, frozen single-normalized DINO and mean/L2. This contains a small executor-supplied target prior; it is not unsupervised. No old cup coordinates or bank were used.',
        '', 'Dataset RGB already follows the converted orientation; no training flip. Raw simulator RGB is flipped once by the official processor. Public conversion code explains10FPS metadata; native one-action/one20Hz-step execution is frozen without interpolation or repetition. Exact historical publisher conversion commit remains unknown.',
        '', 'Object ON/OFF uses the same trained weights and is a mechanism diagnostic, not a fresh training ablation. Failure-stage labels remain UNKNOWN when only the simulator success flag is available; videos and traces are retained.',
        '', 'All checkpoints, optimizer/RNG states, raw logs, trajectories, videos and engineering smoke evidence remain on the server. Compact ZIP excludes large caches/checkpoints and contains unique paths and SHA pointers. No Final15, hardware operation, extra seed/task, HF/Git publication or automatic full-suite expansion.',
        '', 'Independent Builder Review and Red-Team Review are PENDING. The executor checks below do not substitute for these reviews.']
    (OUT/'performance_summary.md').write_text('\n'.join(lines)+'\n')
    (OUT/'summary.md').write_text((OUT/'performance_summary.md').read_text())
    (OUT/'builder_review.md').write_text('# Executor Builder precheck\n\nPASS: actual three10k checkpoints, 27 Dev episode rows,30 paired rollout rows,54 Object ON/OFF rows; prototype/split identity and real100-step optimizer/RNG replay guards. Independent Builder Review PENDING. See raw checks, no claims of real-robot performance.\n')
    (OUT/'redteam_review.md').write_text('# Executor Red-Team precheck\n\nLimits: one task/seed and10 rollouts; small targeted prior, coarse patches with context; camera correction belongs to this fresh round; source conversion commit uncertainty; teacher-forced errors are not success rates; ON/OFF not retraining ablation. First-failure stage unknown without an event oracle. Independent Red-Team Review PENDING. No automatic new experiments.\n')
    write(OUT/'validation.json',dict(status='COMPUTE_DELIVERABLES_VALIDATED',prototype='G2_PROTOTYPE_PASS',
         training='G3_TRAINING_PASS',closed_loop='G4_CLOSED_LOOP_PASS',models=3,updates_each=10000,
         dev9_frames_each=1391,dev_episode_rows=27,paired_rollouts=30,object_diagnostic_rows=54,
         checkpoint_sha_verified=True,pilot_only=True,independent_builder_review='PENDING',independent_redteam_review='PENDING'))
    write(OUT/'final_server_snapshot.json',dict(time=time.time(),gpu=subprocess.getoutput('nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader'),
        disk_free_bytes=shutil.disk_usage(OUT).free,queue_complete=True,training_active=False))
    tools=OUT/'tools';tools.mkdir(exist_ok=True)
    for path in (WORK/'workspace/scripts').glob('*.py'):shutil.copyfile(path,tools/path.name)
    shutil.copyfile(WORK/'workspace/object_conditioning.py',tools/'historical_object_conditioning.py')
    source_files={p.name:sha(p) for p in (WORK/'workspace/scripts').glob('*.py')}
    write(OUT/'code_identity.json',dict(scripts=source_files,object_module_sha256=sha(WORK/'workspace/object_conditioning.py'),
         decision_sha256=sha(WORK/'decision.md'),server_git_head=subprocess.getoutput(f'git -C {ROOT} rev-parse HEAD'),
         dirty_source_explicit_hashes=True,official_stock_commit='8fff0fde7c79f23a93d845d1a50e985de01f8b8a'))
    status.update(status='RESULT_READY',finished=time.time(),training='G3_TRAINING_PASS',closed_loop='G4_CLOSED_LOOP_PASS')
    write(OUT/'status.json',status)
    write(OUT/'active_stage.json',dict(stage='FINISHED',processes=[],last_stage='result packaging'))
    for file in OUT.rglob('*.json'):json.loads(file.read_text())
    manifest={}
    files=[WORK/'decision.md']+[p for p in OUT.rglob('*') if p.is_file() and p.name not in ['artifact_manifest.json','result_package.log'] and p.suffix not in ['.lock','.partial']]
    for file in files:
        packed=file.stat().st_size<10_000_000
        manifest[file.relative_to(WORK).as_posix()]=dict(sha256=sha(file),bytes=file.stat().st_size,packaged=packed,server_path=str(file))
    write(OUT/'artifact_manifest.json',manifest)
    archive=WORK/'P4_8_r5_MATCH_FORMAL6_PAIRED_PILOT_result.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name,item in manifest.items():
            if item['packaged']:z.write(WORK/name,name)
        z.write(OUT/'artifact_manifest.json','result/artifact_manifest.json')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        zip_entries=len(z.namelist())
        for name,item in manifest.items():
            if item['packaged']:
                import hashlib
                assert hashlib.sha256(z.read(name)).hexdigest()==item['sha256']
    write(WORK/'delivery_receipt.json',dict(status='ZIP_CRC_AND_PACKAGED_SHA_PASS',zip_path=str(archive),zip_sha256=sha(archive),
          bytes=archive.stat().st_size,zip_entries=zip_entries,manifest_records=len(manifest),large_files_retained_on_server=True))
    print('R5_RESULT_READY',archive,flush=True)


if __name__=='__main__':main()
