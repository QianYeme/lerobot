"""Authorized Fixed scan queue; conditional branches require semantic evidence review."""
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile

from p410_worker import ROOT,R5,WORK,OUT,old,VALUES
from p410_evaluate import checkpoint
PY='/root/autodl-tmp/p48_isolated_env/bin/python'
SCRIPTS=WORK/'workspace/scripts'

def group(stage,commands):
    ps=[]
    for name,command in commands:
        path=OUT/'logs'/f'{stage}_{name}.log';path.parent.mkdir(exist_ok=True)
        with path.open('a') as f:
            p=subprocess.Popen([PY,'-u']+command,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        ps.append((name,p))
    old.write(OUT/'active_stage.json',dict(stage=stage,pids={n:p.pid for n,p in ps},time=time.time()))
    failures=[]
    for name,p in ps:
        if p.wait():failures.append(name)
    assert not failures,(stage,failures)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'status.json').exists(),'Duplicate launch refused'
    state=dict(status='PREFLIGHT',pid=os.getpid(),started=time.time(),new_train_budget=5000,conditional_branches='NOT_ENTERED')
    old.write(OUT/'status.json',state)
    def merge(paths,name):
        rows=[]
        for p in paths:
            with p.open() as f:
                for r in csv.DictReader(f):r['source_path']=p.relative_to(OUT).as_posix();rows.append(r)
        if rows:
            fields=list(dict.fromkeys(k for r in rows for k in r))
            with (OUT/name).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
        return rows
    try:
        import torch
        sources={}
        for variant in ['G0','G1e-3','NOOBJ']:
            p=checkpoint(variant);s=torch.load(p,map_location='cpu',weights_only=False);assert s['step']==5000
            if variant=='G1e-3':
                assert s['identity']['seed']==1 and s['identity']['kind']=='fixed'
                assert s['identity']['bank_sha256']==old.sha(R5/'result/prototype_bank.pt')
            else:
                assert s['kind']==('act' if variant=='NOOBJ' else 'fixed')
                assert s['split_sha256']==old.sha(R5/'result/dataset_split_manifest.json')
                assert s['stats_sha256']==old.sha(R5/'result/train35_statistics.pt')
            sources[variant]=dict(path=str(p),sha256=old.sha(p),step=5000,independently_trained=variant=='NOOBJ');del s
        old.write(OUT/'execution_manifest.json',dict(reused_checkpoints=sources,seed=1,bank_sha=old.sha(R5/'result/prototype_bank.pt'),
            source_files={p.name:old.sha(p) for p in SCRIPTS.glob('*.py')},environment=subprocess.check_output([PY,'-m','pip','freeze'],text=True),
            protocol=json.loads((R5/'result/timing_contract.json').read_text()),git_head=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
            dataset_revision='a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4',no_final_or_real_robot=True))
        old.write(OUT/'init_contract.json',dict(exploratory_init_ids=list(range(30,50)),seed=1,
            frozen_before_new_training=True,not_blind_final_test=True,heldout_confirmation='NOT_ALLOCATED; remaining states have earlier smoke/usage uncertainty'))
        with (OUT/'gate_sweep_config_matrix.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['variant','gate_initial','steps','reuse'])
            for v,x in [('G0',0),('G1e-3',.001),*VALUES.items(),('NOOBJ','N/A')]:w.writerow([v,x,5000,v in sources])
        # Avoid perturbing the already authorized model-release runtime validation.
        release=ROOT/'experiments/p4_final_p1_training/p4_9_unified_gate_init/index1/release/result/upload/upload_receipts.json'
        state['status']='WAITING_RELEASE_VALIDATION';old.write(OUT/'status.json',state)
        while not release.exists() or len(json.loads(release.read_text()))<2:time.sleep(15)
        variants=list(VALUES)
        group('smoke',[(v,[str(SCRIPTS/'p410_worker.py'),'--variant',v,'--mode','smoke']) for v in variants[:2]])
        group('smoke',[(v,[str(SCRIPTS/'p410_worker.py'),'--variant',v,'--mode','smoke']) for v in variants[2:]])
        state['status']='BENCHMARK';old.write(OUT/'status.json',state)
        records=[];singles={};hashes={};chosen=1;prior_gain=1.
        for n in [1,2,3,4]:
            groups=[[v] for v in variants] if n==1 else [variants[:n]]
            for vs in groups:
                tag='n'+str(n)
                group('bench_'+tag,[(v,[str(SCRIPTS/'p410_worker.py'),'--variant',v,'--mode','bench','--tag',tag]) for v in vs])
                reports=[]
                for v in vs:
                    r=json.loads((OUT/'bench'/v/tag/'complete.json').read_text());reports.append(r)
                    hashes.setdefault(v,r['final_sha']);assert hashes[v]==r['final_sha']
                    records.append(dict(concurrency=n,**r))
                    if n==1:singles[v]=r['timed_seconds']
                gain=sum(singles[v] for v in vs)/max(r['timed_seconds'] for r in reports)
                if n>1 and gain>=prior_gain*1.1:chosen=n;prior_gain=gain
        with (OUT/'parallel_throughput_benchmark.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
        old.write(OUT/'scheduling.json',dict(concurrency=chosen,relative_gain=prior_gain,final_states_identical=True))
        state['status']='TRAINING';old.write(OUT/'status.json',state)
        for i in range(0,len(variants),chosen):group('train',[(v,[str(SCRIPTS/'p410_worker.py'),'--variant',v,'--mode','train']) for v in variants[i:i+chosen]])
        state['status']='EVALUATING';old.write(OUT/'status.json',state)
        all_variants=['G0','G1e-3','NOOBJ']+variants
        for mode in ['dev','rollout']:
            for i in range(0,len(all_variants),min(chosen,3)):
                group(mode,[(v,[str(SCRIPTS/'p410_evaluate.py'),'--variant',v,'--mode',mode]) for v in all_variants[i:i+min(chosen,3)]])
        state['status']='FIXED_SCAN_READY_FOR_SEMANTIC_REVIEW'
        state['conditional_branches']='PENDING_REVIEW_OF_CLOSED_LOOP_AND_TARGET_TRACKING; no optional fixed-gate experiment or formal6 launched'
    except Exception:
        state['status']='FAILED';state['error']=traceback.format_exc()
    old.write(OUT/'status.json',state)
    merge((OUT/'train').glob('*/train_metrics.csv'),'train_metrics.csv')
    merge((OUT/'train').glob('*/checkpoint_manifest.csv'),'checkpoint_manifest.csv')
    merge((OUT/'evaluation').rglob('dev_metrics.csv'),'dev_comparison.csv')
    merge((OUT/'evaluation').rglob('gradients.csv'),'gradients.csv')
    rollouts=merge((OUT/'rollouts').rglob('libero_rollouts.csv'),'rollout_per_init.csv')
    objects=[]
    for p in (OUT/'evaluation').rglob('object_path.json'):
        for r in json.loads(p.read_text()):r['source_path']=p.relative_to(OUT).as_posix();objects.append(r)
    if objects:
        with (OUT/'object_path_diagnostics.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(objects[0]));w.writeheader();w.writerows(objects)
    counts={}
    for r in rollouts:
        variant=r['source_path'].split('/')[1]
        counts.setdefault(variant,[0,0]);counts[variant][0]+=r['success'].lower()=='true';counts[variant][1]+=1
    (OUT/'paired_summary.md').write_text('# Fixed5k paired exploration\n\n'+json.dumps(counts,indent=2)+'\n\nSame20 init30–49, not a blind final test. Results are exploratory, not proof of benefit. Review per-init matched outcomes, video and prototype heatmaps before conditional branches.\n')
    (OUT/'builder_review.md').write_text('# Builder\n\nIndependent PENDING; inspect real artifacts, resume replay, 5k-only pairing, frozen init30–49, sampler and source hashes.\n')
    (OUT/'red_team_review.md').write_text('# Red-Team\n\nIndependent PENDING; single seed/task,20 starts, manually assisted prototypes, no blind-final claim. Full source/runtime preserved; future Legacy/formal6 require evidence of trustworthy mechanism, not larger ON/OFF alone.\n')
    (OUT/'summary.md').write_text('# P4.10 initial Fixed stage\n\n'+json.dumps(state,indent=2)+'\n\nReview heatmaps and matched rollout evidence before conditional branches. No model published or real robot operated.\n')
    with zipfile.ZipFile(WORK/'P4_10_FIXED_STAGE_result.zip','w',zipfile.ZIP_DEFLATED) as z:
        z.write(WORK/'decision.md','decision.md')
        for p in OUT.rglob('*'):
            if p.is_file() and p.suffix not in ['.pt','.pth','.safetensors','.lock']:z.write(p,p.relative_to(WORK).as_posix())

if __name__=='__main__':main()
