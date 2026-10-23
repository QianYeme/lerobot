"""Frozen R5 identities and blocked trial order; no hardware imports."""
import json
from pathlib import Path

from lerobot.utils.f5_protocol import BOUND, POSITION_FIELDS, digest, payload_path

MODELS=['M1','M2','M3','M4','M5']
STAGE_PREFIX='P'
LABELS=dict(zip(MODELS,['Strong-G-36-S3','Strong-G-60-S3','New-GL-36','New-GL-60','G60-BadSeed-S4']))
G36='e122e0e6574cf9e3f1a655b90bdf0de210b769e05dce0663ac451b20aefbd098'
G60='998026c8072aa236cf9dd1c28991c29def968246759fa73c31bdc7415e480c58'
L36='695d6fa89f3e823f6236dca38c8e684c461009d5b9c357cfd1cb9f0d8dffea81'
L60='a71541b5d8d04fd22b8aa1958cd347dd2dced74afd3bd944f6e422d7112d8326'
BAD='1277d98f221a69df2dcfd04361c34828f27799cf4794c25c75a6a83a3a396c8d'
EXPECTED={'M1':(G36,None),'M2':(G60,None),'M3':(G36,L36),'M4':(G60,L60),'M5':(BAD,None)}
SCHEDULE=[(position,model) for position,models in [('A',MODELS),('B',MODELS[1:]+MODELS[:1]),('C',MODELS[2:]+MODELS[:2])] for model in models]


def trial_id(model,position):return f'R5_{model}_{position}_01'


def slot(model,position):return SCHEDULE.index((position,model))+1


def verify_bundle(root,model):
    root=Path(root).resolve();identities=json.loads((root/'model_identity.json').read_text())
    if set(identities)!=set(MODELS):raise ValueError('R5 requires five frozen identities')
    record=identities[model];global_sha,local_sha=EXPECTED[model]
    if (record['global_sha256'],record.get('local_sha256'))!=(global_sha,local_sha):raise ValueError('Wrong R5 frozen model')
    expected_norm='normalization/'+('train36.json' if model in ['M1','M3'] else 'full60.json')
    if record['normalization_path']!=expected_norm:raise ValueError('Wrong normalization family')
    manifest=json.loads((root/'manifest.json').read_text())['sha256']
    names=[record['normalization_path'],'assets/cup_prototype_bank_v1.pt',record['global_path']+'/config.json',
        record['global_path']+'/model.safetensors',record['global_path']+'/f2_1_normalization.json']
    if local_sha:names.append(record['local_path'])
    for name in names:
        if digest(payload_path(root,name))!=manifest.get(name):raise ValueError('R5 payload mismatch: '+name)
    if digest(payload_path(root,record['global_path']+'/model.safetensors'))!=global_sha:raise ValueError('Wrong global weights')
    if payload_path(root,expected_norm).read_bytes()!=payload_path(root,record['global_path']+'/f2_1_normalization.json').read_bytes():
        raise ValueError('Normalizer differs from checkpoint metadata')
    if local_sha and digest(payload_path(root,record['local_path']))!=local_sha:raise ValueError('Wrong local weights')
    return record
