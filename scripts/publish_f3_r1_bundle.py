"""Prepare/upload exactly F3-r1 M0/M1/M2/M3; archive r0 unchanged."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

BASE=Path('/root/autodl-tmp/F3_r1_transfer_20261020')
BUNDLE=BASE/'bundle'
REPO='QYyyyyyyy/F3_R1_MULTI_MODEL_SO101'
HISTORICAL='/root/autodl-tmp/lerobot/results/experiments/T0_POSITION_DISCOVERY_ACT_VS_OCACT/P1_ocfix2_s1/checkpoints/100000/pretrained_model'
P1_REPO='QYyyyyyyy/G1_P1_OCACT_100k'
P1_REV='33a605feb8c29dab284c3aa38740df11d2ae7835'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False))


def prepare(api):
    BUNDLE.mkdir(parents=True,exist_ok=False)
    # Copy accepted r0 migration assets, never modify or delete archived originals.
    shutil.copytree('/root/autodl-tmp/F3_transfer_20261020/bundle',BUNDLE,dirs_exist_ok=True)
    shutil.copy2(BUNDLE/'decision.md',BUNDLE/'decision_r0.md')
    shutil.copy2(BASE/'decision.md',BUNDLE/'decision.md')
    shutil.copy2(BASE/'TRANSFER_GUIDE.md',BUNDLE/'TRANSFER_GUIDE.md')
    shutil.copy2(BASE/'verify_f3_multimodel_bundle.py',BUNDLE/'source/scripts/verify_f3_multimodel_bundle.py')
    m0=Path(HISTORICAL)
    info=api.model_info(P1_REPO,revision=P1_REV,files_metadata=True)
    files={file.rfilename:file for file in info.siblings}
    assert files['model.safetensors'].lfs is not None,'BLOCKED_MODEL_IDENTITY: no pinned weight identity'
    expected=files['model.safetensors'].lfs.sha256
    assert sha(m0/'model.safetensors')==expected,'BLOCKED_MODEL_IDENTITY: M0 weight differs from pinned historical HF model'
    pinned=Path(hf_hub_download(P1_REPO,'config.json',revision=P1_REV,token=os.environ['HF_TOKEN'],headers={'Accept-Encoding':'identity'}))
    assert pinned.read_bytes()==(m0/'config.json').read_bytes(),'BLOCKED_MODEL_IDENTITY: M0 config mismatch'
    shutil.copytree(m0,BUNDLE/'models/M0')
    m2=Path('/root/autodl-tmp/F2_1_r1_20261019/results/A1/checkpoints/060000/pretrained_model')
    assert sha(m2/'model.safetensors')=='27c90813d3e2bba4c9aeba0bd21f41525893eb823e5011f6ced368dd1a7c7a89'
    shutil.copytree(m2,BUNDLE/'models/M2')
    fields=['shoulder_pan.pos','shoulder_lift.pos','elbow_flex.pos','wrist_flex.pos','wrist_roll.pos','gripper.pos']
    identities={}
    for name,path,source in [('M0','models/M0','historical P1_ocfix2_s1 100k'),
        ('M1','global','F2.1 A0 retry1 80k'),('M2','models/M2','F2.1 A1 60k')]:
        cfg=json.loads((BUNDLE/path/'config.json').read_text())
        identities[name]=dict(global_path=path,global_sha256=sha(BUNDLE/path/'model.safetensors'),
            source=source,state_fields=fields+(['gripper.load','gripper.curr'] if name=='M0' else []),
            action_fields=fields,global_config=cfg,cameras=['observation.images.top','observation.images.gripper'],
            separate_load_input=name=='M2',normalization='checkpoint standard processors' if name=='M0' else 'normalization.json',
            project_release_base_git_head='69e28771f4d18994258b1c4540586d1572a72986',
            training_git_head='not recorded; pinned historical HF revision' if name=='M0' else 'a023f783ae752aee740e7beada6d4290a56033d3',
            source_identity='exact payload hashes in manifest.json',common_hardware_configuration='PENDING')
    identities['M0'].update(hf_repo=P1_REPO,hf_revision=P1_REV)
    identities['M3']=dict(identities['M1'],source='same M1 + F2.3 C1-load 10k',local_path='local/010000.pt',
        local_sha256=sha(BUNDLE/'local/010000.pt'),local_normalization='local_normalization.json',
        delta_max=4.832699459791184,local_architecture='3-32-32-1 GELU/tanh',default_mode='disabled')
    save(BUNDLE/'model_identity.json',identities)
    save(BUNDLE/'deployment_config.json',dict(status='PENDING_ON_NEW_SYSTEM',robot='PENDING',cameras='PENDING',
        control_frequency='PENDING',temporal_ensembling='PENDING_COMMON_FOR_ALL_MODELS',action_clipping='PENDING_COMMON',
        smoothing='PENDING_COMMON',task='familiar existing cup task; no expansion',default_local_mode='disabled',
        M3_shadow_required=3,formal_trials_per_model=3,trial_order=['M0','M1','M2','M3']*3,
        M3_delta_max=4.832699459791184,robot_trials='NOT_RUN',F3_R1_COMPLETE=False))
    (BUNDLE/'README.md').write_text('---\nlicense: apache-2.0\ntags:\n- robotics\n- lerobot\n---\n\n'
        '# F3 r1 four-model comparison migration bundle\n\nExactly M0 historical8D P1, M1 clean6D A0, '
        'M2 global single-load A1, M3 same M1 plus local C1-load. See model_identity.json and TRANSFER_GUIDE.md. '
        'Original r0 remains archived. NO robot tests performed. Hardware setup/integration and common deployment parameters pending.\n')
    manifest=dict(repo_id=REPO,models=['M0','M1','M2','M3'],sha256={str(path.relative_to(BUNDLE)):sha(path)
        for path in BUNDLE.rglob('*') if path.is_file() and path.name!='manifest.json'})
    save(BUNDLE/'manifest.json',manifest)
    env=dict(os.environ,PYTHONPATH=str(BUNDLE/'source/src'),TORCH_HOME=str(BUNDLE/'torch_cache'))
    for name in ['verify_f3_bundle.py','verify_f3_multimodel_bundle.py']:
        log=BASE/(name+'.log')
        with log.open('w') as stream:
            subprocess.run(['/root/miniconda3/bin/python',str(BUNDLE/'source/scripts'/name),'--bundle',str(BUNDLE),'--device','cuda'],
                env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        target=BUNDLE/'validation'/(name+'.log');shutil.copy2(log,target)
        manifest['sha256'][str(target.relative_to(BUNDLE))]=sha(target)
    save(BUNDLE/'manifest.json',manifest)
    save(BASE/'model_identity.json',identities)
    print('PREPARED_FOUR_MODEL_OFFLINE_PASS',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['prepare','upload'],required=True);args=parser.parse_args()
    api=HfApi(endpoint='https://huggingface.co',token=os.environ['HF_TOKEN'],headers={'Accept-Encoding':'identity'})
    assert api.whoami()['name']=='QYyyyyyyy'
    if args.mode=='prepare':prepare(api);return
    api.create_repo(REPO,repo_type='model',private=True,exist_ok=True)
    commit=api.upload_folder(repo_id=REPO,repo_type='model',folder_path=str(BUNDLE),commit_message='Publish F3-r1 four-model comparison migration assets; robot trials not run')
    info=api.model_info(REPO,revision=commit.oid,files_metadata=True);files={f.rfilename:f for f in info.siblings}
    manifest=json.loads((BUNDLE/'manifest.json').read_text())
    for name in ['global/model.safetensors','models/M0/model.safetensors','models/M2/model.safetensors','local/010000.pt']:
        lfs=files[name].lfs
        if lfs:assert lfs.sha256==manifest['sha256'][name]
        else:assert sha(Path(hf_hub_download(REPO,name,revision=commit.oid,token=os.environ['HF_TOKEN'],headers={'Accept-Encoding':'identity'})))==manifest['sha256'][name]
    remote=Path(hf_hub_download(REPO,'manifest.json',revision=commit.oid,token=os.environ['HF_TOKEN'],headers={'Accept-Encoding':'identity'}))
    assert remote.read_bytes()==(BUNDLE/'manifest.json').read_bytes()
    save(BASE/'publication.json',dict(repo_id=REPO,revision=commit.oid,private=info.private,models=['M0','M1','M2','M3'],
        all_model_hashes_verified=True,robot_trials='NOT_RUN',common_deployment='PENDING',F3_R1_COMPLETE=False))
    print((BASE/'publication.json').read_text(),flush=True)


if __name__=='__main__':main()
