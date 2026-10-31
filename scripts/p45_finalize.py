"""Watch the bounded P4.5 campaign and aggregate genuine evidence without robot access."""
import csv
import hashlib
import json
from pathlib import Path
import time
import zipfile

import pandas as pd

ROOT=Path('/root/autodl-tmp/lerobot-main')
WORK=ROOT/'experiments/p4_final_p1_training/p4_5_robot_reliability_server10h/index1'
RESULT=WORK/'result';OUT=RESULT/'server'
NAMES=['VACT-S1','VACT-S2','VACT-S3','NG-L-G','NG-F-G','NG-L-L','NG-F-L']


def write(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def summarize():
    runs={};identities=[];checkpoints=[]
    for name in NAMES:
        folder=OUT/'runs'/name;done=folder/'execution_complete.json'
        runs[name]=json.loads(done.read_text()) if done.exists() else dict(status='NOT_COMPLETED')
        identity=folder/'identity.json'
        identities.append(dict(run=name,status=runs[name]['status'],steps=runs[name].get('step',0),
            identity_path=str(identity) if identity.exists() else '',official_stock=name.startswith('VACT'),
            seed=int(name[-1]) if name.startswith('VACT') else 1))
        for path in sorted(folder.rglob('resume.pt'))+sorted(folder.glob('0*.pt')):
            checkpoints.append(dict(run=name,path=str(path),bytes=path.stat().st_size,sha256=sha(path),
                includes_optimizer=True,storage='server retained; not in result zip'))
    pd.DataFrame(identities).to_csv(OUT/'model_run_manifest.csv',index=False)
    pd.DataFrame(checkpoints).to_csv(OUT/'checkpoint_manifest.csv',index=False)
    if (OUT/'dev15_model_comparison.csv').exists():
        frame=pd.read_csv(OUT/'dev15_model_comparison.csv')
        frame.groupby(['run','step','episode']).mean(numeric_only=True).to_csv(OUT/'dev15_per_episode.csv')
        frame.groupby(['run','step']).mean(numeric_only=True).to_csv(OUT/'dev15_model_summary.csv')
    if (OUT/'newgl_dev15_candidates.csv').exists():
        frame=pd.read_csv(OUT/'newgl_dev15_candidates.csv')
        frame.groupby(['run','step']).mean(numeric_only=True).to_csv(OUT/'newgl_global_candidate_summary.csv')
    if (OUT/'newgl_global_local_ablation.csv').exists():
        frame=pd.read_csv(OUT/'newgl_global_local_ablation.csv');selected=[]
        for name in ['NG-L-L','NG-F-L']:
            on=frame[(frame.run==name)&(frame['mode']=='ON')]
            if len(on):
                scores=on.groupby('step').h0_gripper_normalized.mean()
                step=int(scores.sort_index().idxmin());checkpoint=OUT/'runs'/name/f'{step:06d}.pt'
                write(OUT/'runs'/name/'selected_local.json',dict(step=step,primary=float(scores[step]),checkpoint=str(checkpoint),sha256=sha(checkpoint)))
                selected.extend(frame[(frame.run==name)&(frame.step==step)].to_dict('records'))
        if selected:pd.DataFrame(selected).to_csv(OUT/'newgl_selected_global_local_comparison.csv',index=False)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        figures=OUT/'figures';figures.mkdir(exist_ok=True)
        for name in NAMES:
            path=OUT/'runs'/name/'training.csv'
            if not path.exists():continue
            frame=pd.read_csv(path);fig,axes=plt.subplots(1,2,figsize=(10,3))
            for ax,key in zip(axes,['loss','grad_norm']):
                ax.plot(frame.step,frame[key],linewidth=.7);ax.set(xlabel='Update',ylabel=key,title=name)
            fig.tight_layout();fig.savefig(figures/(name+'.png'),dpi=160);plt.close(fig)
    except Exception as error:
        (OUT/'figure_error.log').write_text(repr(error)+'\n')
    complete=all(value['status']=='DONE' for value in runs.values()) and (OUT/'legacy_checkpoint_comparison.csv').exists()
    payload=dict(server='SERVER_COMPLETE' if complete else 'SERVER_PARTIAL',robot='ROBOT_BLOCKED',
        robot_reason='User deferred robot testing; hardware not connected; no robot operation performed',
        runs=runs,finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),final15_accessed=False,
        independent_builder_review='PENDING',independent_red_team_review='PENDING')
    write(RESULT/'execution_complete.json',payload)
    summary='# P4.5 server delivery\n\n'+payload['server']+'\n\n'
    summary+='\n'.join(f'- {name}: {value["status"]}, update {value.get("step","unknown")}' for name,value in runs.items())
    summary+='\n\nRobot tests are deferred by the user. Teacher-forced Dev15 metrics are not robot success rates. '
    summary+='No Final15 model training/evaluation. No interim-score-based extra models. Checkpoint weights remain at exact server paths in checkpoint_manifest.csv.\n'
    (OUT/'server_summary.md').write_text(summary)
    (RESULT/'builder_review.md').write_text('''# Builder self-review; independent review pending

Inspect official stock source manifest and baseline_authenticity.md, real-data stock smoke/reload,
NG preprocessing probe/native schema, complete optimizer/RNG checkpoints, per-run logs and matched L/F budgets.
Check every DONE flag against the actual final update and frozen Dev15 files. Review source hashes and native defaults.
This self-review is not an independent reviewer approval. No robot results are inferred from offline metrics.
''')
    (RESULT/'red_team_review.md').write_text('''# Red-team self-check; independent review pending

Stock6D and P1/ACT1 8D are method baselines with different observations, not a single-variable architecture ablation.
NG Legacy is reconstructed on formal6, not an exact historical formal4 replay. Both branches must finish Global and Local before repair/load claims.
Dev15 uses six fixed progress points per episode, teacher-forced observations; there is no closed-loop reliability evidence.
Historical statement L1 more stable remains subjective unless linked to checkpoint/actual trial records.
Incomplete seeds or stages remain PARTIAL; no success count is filled for deferred robot trials.
''')
    files=[]
    for path in RESULT.rglob('*'):
        if path.is_file() and path.suffix in ['.json','.jsonl','.csv','.md','.log','.png'] and 'checkpoints' not in path.parts and path.name!='delivery_manifest.json':
            files.append(path)
    write(RESULT/'delivery_manifest.json',[dict(path=p.relative_to(WORK).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files])
    with zipfile.ZipFile(WORK/'P4_5_server_result.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in files+[RESULT/'delivery_manifest.json',WORK/'decision.md']:
            archive.write(path,path.relative_to(WORK).as_posix())
    with zipfile.ZipFile(WORK/'P4_5_server_result.zip') as archive:assert archive.testzip() is None
    write(OUT/'delivery_ready.json',dict(status='READY',zip=str(WORK/'P4_5_server_result.zip'),sha256=sha(WORK/'P4_5_server_result.zip'),server=payload['server']))


def main():
    campaign=json.loads((OUT/'campaign_started.json').read_text());pid=campaign['pid']
    while time.time()<campaign['hard_end']:
        cmdline=Path(f'/proc/{pid}/cmdline')
        if not cmdline.exists() or b'p45_controller.py' not in cmdline.read_bytes():break
        time.sleep(30)
    # Avoid racing the controller's initial archive write.
    for _ in range(6):
        cmdline=Path(f'/proc/{pid}/cmdline')
        if not cmdline.exists() or b'p45_controller.py' not in cmdline.read_bytes():break
        time.sleep(5)
    summarize()


if __name__=='__main__':main()
