"""Complete the authorized dose table's arm/gripper max and normalized per-index fields."""
import argparse
import csv
import json

from p48_r6_probe import OUT,R5,Trace,predict,differences
from p48_r5_common import CachedData,make_policy,model_hash


def main(kind):
    import numpy as np
    import torch
    lock=json.loads((OUT/'input_lock.json').read_text());data={s:CachedData(s) for s in ['train','dev']}
    std=data['train'].stats['action']['std'].numpy()+1e-8;result=[]
    with (OUT/f'gate_dose_response_{kind}.csv').open() as f:
        old={(r['checkpoint_step'],r['probe_id'],r['dose_label']):r for r in csv.DictReader(f)}
    for step in [5000,10000]:
        _,policy=make_policy(kind)
        saved=torch.load(R5/'result/runs'/kind/'checkpoints'/f'{step:06d}'/'resume.pt',map_location='cpu',weights_only=False)
        policy.load_state_dict(saved['model'],strict=True);policy.eval();before=model_hash(policy);trace=Trace(policy)
        for sample in lock['samples']:
            batch=data[sample['split']].batch([sample['index']]);off=predict(policy,trace,batch,alpha=0.)
            for label in ['checkpoint',0.,.001,.01,.1,-.01]:
                value=off if label==0 else predict(policy,trace,batch,alpha=None if label=='checkpoint' else label)
                row=dict(old[(str(step),sample['id'],str(label))]);original=dict(row)
                computed=differences(value,off,std)
                assert all(abs(float(original[key])-computed[key])<=1e-12 for key in ['h0_arm_raw','h0_gripper_raw','chunk_normalized_max'])
                error=np.abs(value-off);raw=error*std
                row.update(discrete_normalized_mae=json.dumps({i:error[i].tolist() for i in lock['discrete_chunk_indices']}),
                    chunk_arm_normalized_max=float(error[:,:6].max()),chunk_gripper_normalized_max=float(error[:,6].max()),
                    chunk_arm_raw_max=float(raw[:,:6].max()),chunk_gripper_raw_max=float(raw[:,6].max()))
                result.append(row)
        assert model_hash(policy)==before
        trace.close();del trace,policy,saved
        import gc
        gc.collect();torch.cuda.empty_cache()
    with (OUT/f'gate_dose_response_complete_{kind}.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
    print('COMPLETE_CHUNK_FIELDS_AND_REPRODUCTION_PASS',kind,len(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--kind',choices=['legacy','fixed'],required=True)
    main(parser.parse_args().kind)
