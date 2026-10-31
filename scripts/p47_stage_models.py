"""Stage deployment-only tensors; does not authenticate, publish, or operate hardware."""
import hashlib
import json
from pathlib import Path
import shutil

import torch
from safetensors.torch import save_file, load_file

import p43_export_robot_bundle as historical

ROOT = Path(__file__).resolve().parents[1]
P4 = ROOT/'experiments/p4_final_p1_training'
WORK = P4/'p4_7/index1'
OLD = P4/'p4_3/result/robot/bundle'
SERVER = P4/'p4_5_robot_reliability_server10h/index1/result/server'
DIFF = P4/'p4_5_robot_reliability_server10h/supplements/diffusion_baseline/index1/result'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    identities=json.loads((OLD/'model_identity_manifest.json').read_text())
    jobs=[]
    for role, slug in [('L1','p4-l1-legacy-p1'),('F3','p4-f3-fixed-p1'),('A1','p4-a1-act-matched')]:
        item=identities[role]
        source=Path(item['source_path'])
        assert sha(source)==item['source_resume_sha256']
        jobs.append((role,slug,source,None,item))
    jobs.extend([
        ('VACT-S1','p45-vanilla-act-s1',SERVER/'runs/VACT-S1/checkpoints/080000/resume.pt',None,{}),
        ('VDP-S1','p45-vanilla-diffusion-s1',DIFF/'VDP-S1/checkpoints/100000/resume.pt',None,{}),
        ('NG-L-GL','p45-newgl-legacy-gl',SERVER/'runs/NG-L-G/checkpoints/060000/resume.pt',SERVER/'runs/NG-L-L/010000.pt',{}),
        ('NG-F-GL','p45-newgl-fixed-gl',SERVER/'runs/NG-F-G/checkpoints/100000/resume.pt',SERVER/'runs/NG-F-L/010000.pt',{}),
    ])
    inventory=[]
    for role,slug,source,local,history in jobs:
        dest=WORK/'staging'/role
        dest.mkdir(parents=True,exist_ok=True)
        checkpoint=torch.load(source,map_location='cpu',weights_only=False)
        state={key:value.detach().cpu().contiguous().clone() for key,value in checkpoint['model'].items()}
        save_file(state,str(dest/'model.safetensors'))
        restored=load_file(str(dest/'model.safetensors'))
        assert state.keys()==restored.keys() and all(torch.equal(value,restored[key]) for key,value in state.items())
        config=source.parent/'pretrained_model/config.json'
        if not config.exists():
            config=source.parents[2]/'config/config.json'
        assert config.exists(),str(config)
        shutil.copy2(config,dest/'config.json')
        record=dict(role=role,proposed_repo_id='QYyyyyyyy/lerobot-'+slug,private=True,
                    source_checkpoint_path=str(source),source_checkpoint_sha256=sha(source),
                    step=checkpoint['step'],seed=3 if role=='F3' else 1,
                    model_sha256=sha(dest/'model.safetensors'),tensor_roundtrip_bitwise=True,
                    inference_validation='PENDING_RUNTIME_AND_INDEPENDENT_DOWNLOAD',
                    upload_status='BLOCKED_UPLOAD_NEW_CREDENTIAL_REQUIRED',historical_identity=history)
        if local:
            saved=torch.load(local,map_location='cpu',weights_only=False)
            assert saved['global_sha256']==sha(source) and saved['step']==10000
            save_file({key:value.contiguous().clone() for key,value in saved['model'].items()},str(dest/'local.safetensors'))
            (dest/'local_config.json').write_text(json.dumps(dict(bound=saved['bound'],normalization=saved['normalization'],global_sha256=saved['global_sha256']),indent=2))
            record.update(local_step=10000,local_source_path=str(local),local_source_sha256=sha(local),local_global_binding='PASS')
        # No optimizer/RNG or dataset content leaves the source checkpoints.
        (dest/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
        inventory.append(record)
        print(role,'TENSOR_EXPORT_PASS',flush=True)
        del checkpoint,state,restored
    result=WORK/'result/upload';result.mkdir(parents=True,exist_ok=True)
    (result/'model_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')


if __name__=='__main__':
    main()
