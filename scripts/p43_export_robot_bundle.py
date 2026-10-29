"""Export the four explicitly requested P4.3 candidates; never operates hardware."""
import csv
import hashlib
import json
import random
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
P4 = ROOT/'experiments/p4_final_p1_training'
OUT = P4/'p4_3/result/robot/bundle'
P41 = P4/'p4_1_unattended_p1_pilot/index3/result'
SOURCES = {
    'L1': P41/'checkpoints/pilot/080000/resume.pt',
    'F3': P4/'p4_2_1_dino_repair_parallel/index1/result/workers/F3/runs/F3/checkpoints/formal/080000/resume.pt',
    'A1': P4/'p4_2_unattended_multi_model/index1/result/retry1/ACT_S1_retry1/runs/ACT_S1_retry1/checkpoints/formal/080000/resume.pt',
    'L1_160K': P4/'p4_2_unattended_multi_model/index1/result/runs/P0_P1_S1_160k/checkpoints/formal/160000/resume.pt',
}
FIELDS = ['shoulder_pan.pos','shoulder_lift.pos','elbow_flex.pos','wrist_flex.pos',
          'wrist_roll.pos','gripper.pos','gripper.load','gripper.curr']


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    identities = {}
    for name, source in SOURCES.items():
        source_hash = sha(source)
        if name == 'F3':
            assert source_hash == '92b421820d759b939e04536541d1833eae47d237c1762667eae505a16e75f3f5'
        checkpoint = torch.load(source,map_location='cpu',weights_only=False)
        step = int(checkpoint['step'])
        assert step == (160000 if name == 'L1_160K' else 80000)
        dest = OUT/'models'/name/'pretrained_model'
        dest.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source.parent/'pretrained_model/config.json',dest/'config.json')
        config = json.loads((dest/'config.json').read_text())
        assert config['input_features']['observation.state']['shape'] == [8]
        assert config['output_features']['action']['shape'] == [6]
        state = {k:v.contiguous().clone() for k,v in checkpoint['model'].items()}
        save_file(state,str(dest/'model.safetensors'))
        restored = load_file(str(dest/'model.safetensors'))
        assert set(restored) == set(state)
        assert all(torch.equal(restored[k],v) for k,v in state.items())
        identities[name] = dict(model=name,step=step,seed=1 if name!='F3' else 3,
            source_path=str(source),source_resume_sha256=source_hash,
            pretrained_path=str(dest.relative_to(OUT)),model_sha256=sha(dest/'model.safetensors'),
            config_sha256=sha(dest/'config.json'),state_fields=FIELDS,action_fields=FIELDS[:6],
            normalization='FIXED_SINGLE_NORM' if name=='F3' else 'NO_DINO' if name=='A1' else 'LEGACY_DOUBLE_NORM',
            export_roundtrip_bitwise_equal=True,
            comparison_block='supplement_160k' if name=='L1_160K' else 'original_36')
        print(name,step,source_hash,flush=True)
        del state, restored, checkpoint
    assets = OUT/'assets'; assets.mkdir(exist_ok=True)
    for name in ['train90_statistics.pt','formal6_train90_prototypes.pt','prototype_provenance.json']:
        shutil.copy2(P41/name,assets/name)
    for identity in identities.values():
        identity['statistics_sha256'] = sha(assets/'train90_statistics.pt')
        identity['prototype_sha256'] = sha(assets/'formal6_train90_prototypes.pt')
    (OUT/'model_identity_manifest.json').write_text(json.dumps(identities,indent=2)+'\n')
    hub = OUT/'torch_cache/hub'; hub.mkdir(parents=True,exist_ok=True)
    shutil.copytree(Path.home()/'.cache/torch/hub/facebookresearch_dinov2_main',
                    hub/'facebookresearch_dinov2_main',dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('.git','__pycache__'))
    (hub/'checkpoints').mkdir(exist_ok=True)
    for name in ['dinov2_vits14_pretrain.pth','resnet18-f37072fd.pth']:
        shutil.copy2(Path.home()/'.cache/torch/hub/checkpoints'/name,hub/'checkpoints'/name)
    # Preserve the original 36-trial block, then add a separately labelled 12-trial supplement.
    rng = random.Random(20261029); rows=[]
    for repeat in [1,2]:
        for pos in range(1,7):
            models=['L1','F3','A1'];rng.shuffle(models)
            for model in models:
                rows.append(dict(slot=len(rows)+1,position=f'P{pos:02d}',repeat=repeat,model=model,block='original_36'))
    for repeat in [1,2]:
        for pos in range(1,7):
            rows.append(dict(slot=len(rows)+1,position=f'P{pos:02d}',repeat=repeat,model='L1_160K',block='supplement_160k'))
    with (OUT/'trial_plan_frozen.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('BUNDLE',OUT,flush=True)


if __name__=='__main__':
    main()
