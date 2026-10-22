"""F5 identities and trial metadata. No robot, tensor or camera imports."""
import hashlib
import json
from pathlib import Path

POSITION_FIELDS=['shoulder_pan.pos','shoulder_lift.pos','elbow_flex.pos','wrist_flex.pos','wrist_roll.pos','gripper.pos']
MODELS=['P1-Hist','Strong-G','L-Load','New-GL']
TOKENS=dict(zip(MODELS,['P1Hist','StrongG','LLoad','NewGL']))
BOUND=4.832699459791184
EXPECTED={
    'P1-Hist':('0b0270f6a0c0d37a23f2a8f5a50be3cddd39f923f9159f4576ad39761c805594',None),
    'Strong-G':('e122e0e6574cf9e3f1a655b90bdf0de210b769e05dce0663ac451b20aefbd098',None),
    'L-Load':('65be80d1a70c178cffd29b8efed8ab08d923a79013eb56bdaa3475e69679fdcf','a6426e26f6c63eb541a572095155a553f0cd78725c8d452ccc1c8637810e8de6'),
    'New-GL':('e122e0e6574cf9e3f1a655b90bdf0de210b769e05dce0663ac451b20aefbd098','695d6fa89f3e823f6236dca38c8e684c461009d5b9c357cfd1cb9f0d8dffea81')}
SCHEDULE=[(position,model) for position,models in [('A',MODELS),('B',MODELS[1:]+MODELS[:1]),('C',MODELS[2:]+MODELS[:2])] for model in models]


def trial_id(model,position):return f'F5_{TOKENS[model]}_{position}_01'


def slot(model,position):return SCHEDULE.index((position,model))+1


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def payload_path(root,name):
    root=Path(root).resolve();path=(root/name).resolve()
    if not path.is_relative_to(root):raise ValueError('Bundle path escapes its root')
    return path


def verify_bundle(root,model):
    root=Path(root).resolve();identities=json.loads((root/'model_identity.json').read_text())
    if set(identities)!=set(MODELS):raise ValueError('F5 requires exactly four frozen model identities')
    record=identities[model];expected=EXPECTED[model]
    if record['global_sha256']!=expected[0] or record.get('local_sha256')!=expected[1]:raise ValueError('F5 checkpoint differs from decision')
    manifest=json.loads((root/'manifest.json').read_text())['sha256']
    names=['normalization.json','assets/cup_prototype_bank_v1.pt',record['global_path']+'/config.json',record['global_path']+'/model.safetensors']
    if expected[1]:names.append(record['local_path'])
    if model=='P1-Hist':names.extend(n for n in manifest if n.startswith(record['global_path']+'/policy_'))
    for name in names:
        actual=digest(payload_path(root,name))
        if actual!=manifest.get(name):raise ValueError(f'Payload identity mismatch: {name}')
    if digest(payload_path(root,record['global_path']+'/model.safetensors'))!=expected[0]:raise ValueError('Wrong global weights')
    if expected[1] and digest(payload_path(root,record['local_path']))!=expected[1]:raise ValueError('Wrong local weights')
    return record
