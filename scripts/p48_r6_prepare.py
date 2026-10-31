"""Lock existing r5 assets and deterministic observations; no policy updates or asset writes."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path('/root/autodl-tmp/lerobot-main')
R5=ROOT/'experiments/p4_public_benchmark/p4_8_libero_pilot/index5'
OUT=R5.parent/'index6/result'
sys.path.insert(0,str(R5/'workspace/scripts'))
from p48_r5_common import CachedData,sha,write


def main():
    import numpy as np
    import torch
    snapshot=dict(time=subprocess.getoutput('date -Is'),gpu=subprocess.getoutput('nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader'),
                  active_jobs=subprocess.getoutput('ps -eo pid,comm,args | grep -E "python.*(p48_r5_worker|p48_r5_evaluate|p45_.*worker|p47_)" | grep -v grep'))
    assert not snapshot['active_jobs'],'Do not interfere with an existing formal job'
    records=[]
    with (R5/'result/checkpoint_manifest.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==6
    for row in rows:
        path=Path(row['path']);assert path.exists() and sha(path)==row['sha256'],path
        state=torch.load(path,map_location='cpu',weights_only=False)
        assert state['kind']==row['kind'] and state['step']==int(row['step'])
        assert state['split_sha256']==sha(R5/'result/dataset_split_manifest.json')
        assert state['stats_sha256']==sha(R5/'result/train35_statistics.pt')
        if row['kind']!='act':assert state['bank_sha256']==sha(R5/'result/prototype_bank.pt')
        records.append(dict(**row,external_config_path=str(R5/'result/effective_configs'/f'{row["kind"]}.yaml'),
            external_config_sha256=sha(R5/'result/effective_configs'/f'{row["kind"]}.yaml'),
            embedded_config_available=False,metadata_verified=True))
        del state
    write(OUT/'upstream_manifest.json',dict(checkpoints=records,snapshot=snapshot,
        prototype_path=str(R5/'result/prototype_bank.pt'),prototype_sha256=sha(R5/'result/prototype_bank.pt'),
        split_sha256=sha(R5/'result/dataset_split_manifest.json'),stats_sha256=sha(R5/'result/train35_statistics.pt'),
        source_files={str(p):sha(p) for p in list((R5/'workspace/scripts').glob('*.py'))+[R5/'workspace/object_conditioning.py']},
        official_act_commit='8fff0fde7c79f23a93d845d1a50e985de01f8b8a',server_git_head=subprocess.getoutput(f'git -C {ROOT} rev-parse HEAD')))
    samples=[];datasets={s:CachedData(s) for s in ['dev','train']}
    for split,data in datasets.items():
        ids=sorted(set(data.arrays['episode'].tolist()))
        if split=='train':ids=[ids[0],ids[len(ids)//2],ids[-1]]
        for ep in ids:
            indices=np.flatnonzero(data.arrays['episode']==ep)
            for phase,index in zip(['start','middle','end'],[indices[0],indices[len(indices)//2],indices[-1]]):
                samples.append(dict(split=split,episode=int(ep),frame=int(data.arrays['frame'][index]),index=int(index),role=phase))
    for r in json.loads((R5/'result/prototype_sources.json').read_text()):
        data=datasets['train'];index=int(np.flatnonzero((data.arrays['episode']==r['episode'])&(data.arrays['frame']==r['frame']))[0])
        samples.append(dict(split='train',episode=r['episode'],frame=r['frame'],index=index,role='bank_reference'))
    for item in samples:
        data=datasets[item['split']];index=item['index']
        item['id']=f'{item["split"]}_ep{item["episode"]}_f{item["frame"]}'
        item['rgb_sha256']=[hashlib.sha256(np.asarray(data.images[index,c]).transpose(1,2,0).tobytes()).hexdigest() for c in range(2)]
        item['state_sha256']=hashlib.sha256(data.arrays['state'][index].tobytes()).hexdigest()
    lock=dict(samples=samples,dev_rule='All9 sorted Dev episodes, start/middle/end=27',
        train_rule='First/median/last sorted Train episodes, same three phases=9; plus four original bank references',
        gradient_train_ids=[s['id'] for s in samples if s['split']=='train' and s['role']!='bank_reference'][:8],
        gradient_dev_ids=[s['id'] for s in samples if s['split']=='dev'][:8],
        dependency_probe_ids=[s['id'] for s in samples if s['episode'] in [857,813] and s['role']!='bank_reference'],
        gate_doses=['checkpoint',0,.001,.01,.1,-.01],discrete_chunk_indices=[0,1,5,10,25,50,99],
        dtype='float32, deterministic, TF32off; same as r5 training numerical mode',
        precision_modes_equivalent=True,permutation_seed=1,random_train_patch_seed=1,
        permutation_rule='PCG64 seed1 permutation; deterministic left rotation if identity',
        random_train_patch_rule='PCG64 seed(1+source_episode), exclude reference ROI patch indices123/124; semantic unrelatedness not guaranteed',
        no_final_data=True,new_rollouts=0)
    path=OUT/'input_lock.json'
    if path.exists():assert json.loads(path.read_text())==lock
    else:write(path,lock)
    print('R6_UPSTREAM_AND_INPUT_LOCK_PASS',len(samples),flush=True)


if __name__=='__main__':main()
