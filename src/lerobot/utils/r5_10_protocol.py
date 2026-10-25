"""Frozen R5.10 model identities, stage semantics and physical start-state audit."""
import json
import math
from pathlib import Path

from lerobot.utils.f5_protocol import POSITION_FIELDS,digest,payload_path

MODELS=['M1','M2','M3','M4']
PRIMARY_MODELS=MODELS[:3]
LABELS={'M1':'C10-S3-80k','M2':'C5-S2-80k','M3':'C5-S2-SWA_60_80','M4':'C5-S3-SWA_60_80_optional'}
EXPECTED={'M1':('clip10',3,[80000]),'M2':('clip5',2,[80000]),'M3':('clip5',2,[60000,80000]),'M4':('clip5',3,[60000,80000])}
SCHEDULE=[(p,m) for p in ['A','B','C'] for m in PRIMARY_MODELS]+[(p,'M4') for p in ['A','B','C']]
STAGE_PREFIX='P'
STAGE_NAMES={'P0':'target-directed approach','P1':'valid grasp geometry','P2':'close command / closure attempt',
    'P3':'physical grasp established','P4':'cup lifted','P5':'hold achieved','P6':'place / release completed'}


def trial_id(model,position):return f'R5_10_{model}_{position}_01'


def slot(model,position):return SCHEDULE.index((position,model))+1


def verify_bundle(root,model):
    root=Path(root).resolve();identities=json.loads((root/'model_identity.json').read_text())
    if set(identities)!=set(MODELS):raise ValueError('R5.10 requires three primary identities and one explicitly optional identity')
    record=identities[model]
    if (record['clip_recipe'],record['seed'],record['source_steps'])!=EXPECTED[model]:raise ValueError('Wrong frozen R5.10 identity')
    if record.get('local_path') or record.get('local_sha256'):raise ValueError('R5.10 global-only models have no local regulator')
    if record['normalization_path']!='normalization/full60.json':raise ValueError('All R5.10 models require Full60 normalization')
    manifest=json.loads((root/'manifest.json').read_text())['sha256']
    names=[record['global_path']+'/model.safetensors',record['global_path']+'/config.json',
        record['global_path']+'/f2_1_normalization.json',record['normalization_path'],'assets/cup_prototype_bank_v1.pt']
    for name in names:
        if digest(payload_path(root,name))!=manifest.get(name):raise ValueError('Payload corrupted or wrong model: '+name)
    if digest(payload_path(root,record['global_path']+'/model.safetensors'))!=record['global_sha256']:raise ValueError('Wrong checkpoint weights')
    if payload_path(root,record['normalization_path']).read_bytes()!=payload_path(root,record['global_path']+'/f2_1_normalization.json').read_bytes():raise ValueError('Normalizer differs from checkpoint')
    return record


def audit_initial(observation,args,common):
    state={k:float(observation[k]) for k in POSITION_FIELDS}
    load=float(observation['gripper.load'])
    if not all(math.isfinite(v) for v in [*state.values(),load]):
        raise ValueError('Nonfinite initial joint state or gripper.load')
    return dict(valid=None,status='RECORDED_MANUAL_RESET',
        state=state,gripper_position=state['gripper.pos'],load=load,
        start_method='operator manually restores starting pose; no numeric Home/tolerance check',
        MODEL_ID=args.model,POSITION=args.position,robot_id=args.robot_id,
        camera_evidence='Same-trial top/gripper_first.jpg captured before first autonomous action')
