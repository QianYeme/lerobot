"""R5.11 identities and manual-start evidence; no physical stage decisions."""
import json
from pathlib import Path
from lerobot.utils.f5_protocol import POSITION_FIELDS,digest,payload_path
from lerobot.utils.r5_10_protocol import audit_initial,STAGE_NAMES

MODELS=['R0','R1','R2','R3']
STAGE_PREFIX='P'
SCHEDULE=[('A',m) for m in MODELS]+[(p,m) for p in ['B','C'] for m in MODELS[1:]]
R3_SHA='d094de659f58df3e55cd6284a83729cfd8658fb1d4b19f73d6a8cb634a61fc0f'
FULL60_NORM_SHA='32f8ebea2aee156b8eabad78f67389278bae0c8e556f6f9370271070d9a780ab'
LABELS={'R0':'P1-Hist','R1':'Train36 New-GL','R2':'R5.10 M1 clip10 seed3 80k','R3':'R5.3 historical Full60 seed3 133k'}


def trial_id(model,position):return f'R5_11_{model}_{position}_01'


def slot(model,position):return SCHEDULE.index((position,model))+1


def verify_bundle(root,model='R3'):
    root=Path(root).resolve()
    if model!='R3':raise ValueError('This bundle contains only R3; reuse existing reference bundles')
    identities=json.loads((root/'model_identity.json').read_text())
    if set(identities)!={'R3'}:raise ValueError('Only the historical133k model may be published here')
    record=identities['R3']
    if record['global_sha256']!=R3_SHA or record.get('local_path'):raise ValueError('Not the historical R5.3 global-only133k')
    if record['source_step']!=133000 or record['normalization_path']!='normalization/full60.json':raise ValueError('Wrong step or normalization')
    manifest=json.loads((root/'manifest.json').read_text())['sha256']
    for name in [record['global_path']+'/model.safetensors',record['global_path']+'/config.json',record['global_path']+'/f2_1_normalization.json',record['normalization_path'],'assets/cup_prototype_bank_v1.pt']:
        if digest(payload_path(root,name))!=manifest.get(name):raise ValueError('Corrupted payload: '+name)
    if digest(payload_path(root,record['global_path']+'/model.safetensors'))!=R3_SHA:raise ValueError('Wrong R3 weights')
    if digest(payload_path(root,record['normalization_path']))!=FULL60_NORM_SHA:raise ValueError('Wrong Full60 normalizer')
    if payload_path(root,record['normalization_path']).read_bytes()!=payload_path(root,record['global_path']+'/f2_1_normalization.json').read_bytes():raise ValueError('Checkpoint normalizer mismatch')
    return record
