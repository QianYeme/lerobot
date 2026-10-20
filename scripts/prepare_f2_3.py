"""Freeze F2.3 identities, Train-only bound and matched lightweight-MLP budget."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/force/f2_3_global_vs_local_load_regulation/index1'
PARENT=ROOT/'experiments/force/f2_1_load_temporal_encoding_ablation/index1/result'


def residual_target(demo, nominal, bound):
    return np.clip(np.asarray(demo)-np.asarray(nominal),-bound,bound)


def main():
    out=BASE/'result'; out.mkdir(parents=True,exist_ok=True)
    assert not (out/'residual_config.json').exists(), 'Do not refreeze an existing experiment'
    split=json.loads((PARENT/'split_manifest.json').read_text())
    arrays=Path(json.loads((PARENT/'r1_preparation.json').read_text())['raw'])
    actions=[]
    for ep in split['train']:
        with np.load(arrays/f'ep{ep:03d}.npz') as z:
            actions.append(z['action'][:,5].copy())
    values=np.concatenate(actions)
    low,high=float(values.min()),float(values.max())
    assert np.isfinite(values).all() and high>low and len(values)==17243
    config=dict(delta_max=.05*(high-low),train_action_min=low,train_action_max=high,
        bound_formula='.05 * (Train36 demo gripper.max - demo gripper.min)',
        bound_units='native demonstrated gripper position units, not Newtons',
        local_inputs={'C1-no-load':['gripper.pos[t]','frozen_global_nominal_gripper[t]'],
                      'C1-load':['normalized_load[t]','gripper.pos[t]','frozen_global_nominal_gripper[t]']},
        mlp_hidden=[32,32],activation='GELU',output='delta_max*tanh(last_linear)',
        pos_nominal_normalization='Train36 mean/std ddof0, std floor1e-8',
        load_normalization='reuse frozen F2.1 Train-only mean/std',
        target='clip(demo-native_nominal,-delta_max,+delta_max)/delta_max; MSE in bound-normalized units',
        optimizer='Adam',learning_rate=.001,batch_size=256,steps=10000,seed=1,
        checkpoint_every=1000,selection='lowest Dev12 frame-weighted normalized h0 gripper L1; tie earlier',
        initialization='seed1 input projection; seed2 common hidden; zero final layer; independent fresh models',
        sampling='same seeded torch.randint Train index sequence, with replacement',
        local_rate='same per-observation rate as global replay; no high-frequency control',
        near_bound_definition='abs(predicted_delta) >= .95*delta_max',
        frozen_utc=datetime.now(timezone.utc).isoformat(),
        contract_sha256=hashlib.sha256((BASE/'decision_F2_3_GLOBAL_VS_LOCAL_LOAD_REGULATION.md').read_bytes()).hexdigest(),
        final_sealed=True,train_episodes=split['train'])
    identity=dict(C0=dict(parent='F2.1-r1 A1',step=60000,primary=.12639014509042482,
        checkpoint='/root/autodl-tmp/F2_1_r1_20261019/results/A1/checkpoints/060000/pretrained_model',
        model_sha256='27c90813d3e2bba4c9aeba0bd21f41525893eb823e5011f6ced368dd1a7c7a89'),
        C1_global=dict(parent='F2.1-r1 A0 retry1',step=80000,primary=.1503186063577208,
        checkpoint='/root/autodl-tmp/F2_1_r1_20261019/results_A0_retry1/A0/checkpoints/080000/pretrained_model',
        model_sha256='65be80d1a70c178cffd29b8efed8ab08d923a79013eb56bdaa3475e69679fdcf',
        historical_caveat='original A0 numerical failure and one authorized retry disclosed in F2.1'),
        git_HEAD=__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        C0_retraining=False,C1_global_frozen=True)
    for name,value in [('residual_config.json',config),('baseline_identity.json',identity)]:
        (out/name).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(config,indent=2))


if __name__=='__main__':main()
