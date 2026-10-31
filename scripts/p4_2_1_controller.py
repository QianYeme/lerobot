"""Measured repair scheduling alongside an untouched legacy queue; bounded delivery."""
import csv
import hashlib
import json
import os
import shutil
import subprocess
import time
import zipfile
from collections import deque
from pathlib import Path

ROOT=Path('/root/autodl-tmp/lerobot-main')
WORK=ROOT/'experiments/p4_final_p1_training/p4_2_1_dino_repair_parallel/index1'
OUT=WORK/'result'
CODE=WORK/'code'
PY='/root/miniconda3/bin/python'
ENV=dict(os.environ,PYTHONPATH=str(CODE/'src')+':'+str(CODE/'scripts'),PYTHONNOUSERSITE='1')


def write(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2)+'\n')


def row(name,value):
    path=OUT/name
    present=path.exists()
    with path.open('a',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(value))
        if not present:writer.writeheader()
        writer.writerow(value)


def start(arguments,label):
    stream=(OUT/(label+'.log')).open('w')
    process=subprocess.Popen([PY,str(CODE/'scripts/p4_2_1_repair.py')]+arguments,env=ENV,cwd=ROOT,
        stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    stream.close()
    return process


def legacy_sample():
    base=ROOT/'experiments/p4_final_p1_training/p4_2_unattended_multi_model/index1/result/runs'
    files=list(base.glob('*/training_metrics.csv'))
    if not files:return None
    path=max(files,key=lambda p:p.stat().st_mtime)
    with path.open() as stream:
        last=list(deque(stream,maxlen=2))[-1].strip().split(',')
    if not last or last[0]=='step':return None
    return str(path),int(last[0]),float(last[5])


def legacy_rate(before,after):
    if before and after and before[0]==after[0] and after[2]>before[2]:
        return (after[1]-before[1])/(after[2]-before[2])
    return None


def resource():
    text=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu','--format=csv,noheader,nounits']).decode().strip()
    memory,total,gpu,power,temp=[float(x.strip()) for x in text.split(',')]
    row('resource_utilization.csv',dict(time_unix=time.time(),memory_used_mib=memory,memory_total_mib=total,
        utilization=gpu,power_w=power,temperature_c=temp,load_average=os.getloadavg()[0]))
    return memory,total


def wait(processes,deadline):
    peak=0
    while any(p.poll() is None for p in processes):
        memory,total=resource()
        peak=max(peak,memory)
        if time.monotonic()>=deadline:
            for process in processes:
                if process.poll() is None:process.terminate()
            return False,peak/total
        time.sleep(2)
    return all(p.returncode==0 for p in processes),peak/total


def benchmark(names,seeds,deadline):
    before=legacy_sample()
    processes=[start(['benchmark','--name',n,'--seed',str(s)],'bench_'+n) for n,s in zip(names,seeds)]
    ok,used=wait(processes,deadline)
    after=legacy_sample()
    values=[]
    for name in names:
        path=OUT/'benchmarks'/name/'result.json'
        if path.exists():values.append(json.loads(path.read_text()))
    rate=legacy_rate(before,after)
    for name,value in zip(names,values):
        row('parallel_benchmark.csv',dict(label=name,candidate_processes=len(names),
            legacy_active=before is not None,legacy_steps_per_s=rate,candidate_steps_per_s=value['steps_per_s'],
            gpu_used_fraction=used,success=ok,reference_scope='candidate single-process tests include untouched live legacy background'))
    return ok,values,used,rate


def package():
    import pandas as pd
    for source_name,target in [('dev15_model_comparison.csv','dev15_matched_comparison.csv'),
                               ('p1_object_path_diagnostics.csv','object_path_diagnostics.csv')]:
        pieces=[]
        if (OUT/source_name).exists():pieces.append(pd.read_csv(OUT/source_name))
        for p in (OUT/'workers').glob('*/'+source_name):pieces.append(pd.read_csv(p))
        if pieces:pd.concat(pieces,ignore_index=True).to_csv(OUT/target,index=False)
    for source_name,target in [('run_manifest.csv','training_runs_manifest.csv'),('checkpoint_manifest.csv','checkpoint_manifest.csv')]:
        pieces=[pd.read_csv(p) for p in (OUT/'workers').glob('*/'+source_name)]
        if pieces:pd.concat(pieces,ignore_index=True).to_csv(OUT/target,index=False)
    curves=[]
    for p in (OUT/'workers').glob('*/runs/*/training_metrics.csv'):
        frame=pd.read_csv(p);frame.insert(0,'run',p.parent.name);curves.append(frame)
    if curves:pd.concat(curves,ignore_index=True).to_csv(OUT/'training_curves.csv',index=False)
    primary=False
    label='COMPARISON_INCONCLUSIVE'
    path=OUT/'dev15_matched_comparison.csv'
    if path.exists():
        data=pd.read_csv(path)
        means=data.groupby(['run','step'])[['arm_normalized','gripper_normalized','arm_raw_mae','gripper_raw_mae']].mean()
        means.to_csv(OUT/'dev15_summary.csv')
        counts=data.groupby(['run','step']).size()
        episodes=data.groupby(['run','step']).episode.nunique()
        primary=all((name,80000) in means.index and counts.loc[(name,80000)]==90 and episodes.loc[(name,80000)]==15 for name in ('L1','F1'))
        if primary:
            old=means.loc[('L1',80000)];fixed=means.loc[('F1',80000)]
            label='REPAIR_BENEFIT_SUPPORTED' if all(fixed[['arm_normalized','gripper_normalized']]<old[['arm_normalized','gripper_normalized']]) else 'REPAIR_WORSE' if all(fixed[['arm_normalized','gripper_normalized']]>old[['arm_normalized','gripper_normalized']]) else 'REPAIR_NO_CLEAR_GAIN'
    diagnostic=OUT/'object_path_signed_diagnostics.csv'
    if diagnostic.exists():
        frame=pd.read_csv(diagnostic)
        frame.groupby(['run','episode'])[['chunk_mean_abs_on_off','h0_pan_signed_on_off','h0_gripper_signed_on_off']].agg(['mean','median',lambda x:x.quantile(.05),lambda x:x.quantile(.95)]).to_csv(OUT/'object_path_per_episode_quantiles.csv')
    failures=list(OUT.glob('*failure*'))+[p for p in OUT.glob('*.log') if 'Traceback' in p.read_text(errors='replace')]
    docs={
        'summary.md':f'# DINO repair result\n\nOutcome={label}\nPRIMARY_MATCHED_EVALUATION_AVAILABLE={primary}\nP4_2_1_PRIMARY_COMPLETE=false (independent reviews pending)\nNo real robot; Final15 not used. Metrics are fixed Dev progress-proxy samples, not physical task success. See scheduler_plan_actual.csv for actual complete/partial/skipped runs.\n',
        'validation.md':'# Gates\n\nengineering_gate.json and preprocess_equivalence.csv record actual DINO prebackbone/feature matching on Train ep0/29/58 and ResNet tensor invariance, plus100-step smoke/restore. Source isolation and raw Top are explicit. Missing evidence is not PASS.\n',
        'source_and_checkpoint_identity.md':'# Identity\n\nSource patch/hash and original source hash in isolation_manifest.json/code_patch.diff. Fixed runs are fresh step0, not resumed from legacy. Train statistics and prototype bytes reused exactly, privately copied per run. Current P4.2 source/config/weights are unchanged. Historical L1 initialization is reconstructed by seed/source; its original step0 checkpoint was not persisted, so recipe comparability retains that declared limitation.\n',
        'smoke_test.md':'# Smoke\n\nSee engineering_gate.json and smoke_metrics.csv.100 optimizer steps are disposable and not added to formal trajectory; checkpoint independently loaded into a new model instance, same-input outputs compared exactly.\n',
        'failure_report.md':'# Failures\n\n'+('\n'.join(str(p) for p in failures) if failures else 'NONE')+'\n',
        'builder_review.md':'# Builder review\n\nExecutor evidence only; independent review PENDING. Check preprocessing <=1e-5, ResNet <=1e-6, Train-only sources, fresh initialization, source isolation, benchmark context, hashes and same-sample Dev.\n',
        'red_team_review.md':'# Red-Team\n\nIndependent review PENDING. Does input consistency translate into useful localization rather than only MAE? Legacy and fixed may share live GPU contention; throughput reference context is disclosed. No strict full-trajectory determinism claim. Check source/init/data exposure equality and no hotfix of legacy.\n',
        'PROPOSAL.md':'# Next decision\n\nChoose2–3 candidates for minimal same-scene real-robot validation only after review. Do not assume repaired P1 is best. New-GL not run: historical6D+load global plus separately trained bounded local10k semantics are not a single unambiguous80k8D model; requires a distinct reconstruction contract. No extra Force module authorized.\n',
    }
    for name,text in docs.items():(OUT/name).write_text(text)
    delivery=WORK/'delivery_small';delivery.mkdir(exist_ok=True)
    for path in OUT.iterdir():
        if path.is_file() and path.stat().st_size<30_000_000:shutil.copyfile(path,delivery/path.name)
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in delivery.iterdir() if p.is_file() and p.name!='delivery_manifest.json'}
    (delivery/'delivery_manifest.json').write_text(json.dumps(hashes,indent=2))
    with zipfile.ZipFile(WORK/'P4_2_1_r0_result_delivery.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in delivery.iterdir():z.write(p,'result/'+p.name)
    with zipfile.ZipFile(WORK/'P4_2_1_r0_result_delivery.zip') as z:assert z.testzip() is None
    print('REPAIR_DELIVERY_READY',label,flush=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'controller_started.json').exists(),'DUPLICATE_CONTROLLER'
    started=time.monotonic();cutoff=started+7.5*3600;deadline=started+9*3600
    write('controller_started.json',dict(start_unix=time.time(),training_cutoff_monotonic=cutoff,
        finish_deadline_monotonic=deadline,max_s=10*3600,legacy_untouched=True))
    process=start(['preflight'],'preflight')
    ok,_=wait([process],min(deadline,started+25*60))
    if not ok:
        write('preflight_failure.json',dict(status='BLOCKED',returncode=process.returncode));package();return
    gate=json.loads((OUT/'engineering_gate.json').read_text());assert gate['status']=='PASS'
    before=legacy_sample();time.sleep(30);base_rate=legacy_rate(before,legacy_sample())
    a,one,used1,legacy1=benchmark(['s1'],[1],deadline)
    b,two,used2,legacy2=benchmark(['s2'],[2],deadline)
    c,dual,used3,legacy3=benchmark(['dual1','dual2'],[1,2],deadline)
    degree=1
    if a and b and c and len(dual)==2 and legacy1 is not None and legacy2 is not None and legacy3 is not None:
        baseline=max(one[0]['steps_per_s']+legacy1,two[0]['steps_per_s']+legacy2)
        total=sum(x['steps_per_s'] for x in dual)+legacy3
        own=all(dual[i]['steps_per_s']>=.55*[one[0],two[0]][i]['steps_per_s'] for i in range(2))
        if total>=1.10*baseline and own and used3<=.85 and (base_rate is None or legacy3>=.55*base_rate):degree=2
    write('parallel_selection.json',dict(fixed_processes=degree,legacy_processes_counted=True,
        no_three_fixed_test='would add a fourth GPU process while legacy is active; two sufficient or failed gate',
        legacy_alone_reference_steps_per_s=base_rate,benchmark_reference='single fixed candidate under untouched legacy background; compare whole-host aggregate too'))
    if base_rate and legacy1 is not None and one and one[0]['steps_per_s']+legacy1<1.10*base_rate:
        row('scheduler_plan_actual.csv',dict(time_unix=time.time(),run='WAIT_LEGACY',status='AGGREGATE_NOT_BENEFICIAL',pid=0))
        legacy_pid=31212
        while time.monotonic()<cutoff:
            try:os.kill(legacy_pid,0)
            except ProcessLookupError:break
            time.sleep(15)
    running=[];pending=[('F1',1),('F2',2),('F3',3)]
    while (pending or running) and time.monotonic()<cutoff+1800:
        while pending and len(running)<degree and time.monotonic()<cutoff:
            name,seed=pending.pop(0)
            process=start(['train','--name',name,'--seed',str(seed),'--cutoff',str(cutoff)],name)
            running.append((name,process));row('scheduler_plan_actual.csv',dict(time_unix=time.time(),run=name,status='STARTED',pid=process.pid))
        remaining=[]
        for name,process in running:
            if process.poll() is None:remaining.append((name,process))
            else:row('scheduler_plan_actual.csv',dict(time_unix=time.time(),run=name,status='EXIT_'+str(process.returncode),pid=process.pid))
        running=remaining
        if not running and (not pending or time.monotonic()>=cutoff):break
        resource();time.sleep(5)
    for name,process in running:
        if process.poll() is None:process.terminate()
        row('scheduler_plan_actual.csv',dict(time_unix=time.time(),run=name,status='SAFETY_TERMINATED_LAST_PERIODIC_CHECKPOINT_RETAINED',pid=process.pid))
    for name,_ in pending:row('scheduler_plan_actual.csv',dict(time_unix=time.time(),run=name,status='NOT_RUN_TIME',pid=0))
    process=start(['evaluate'],'matched_legacy_evaluation')
    wait([process],deadline-15*60)
    package()


if __name__=='__main__':
    main()
