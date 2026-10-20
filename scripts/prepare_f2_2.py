"""Freeze F2.2 Train-only gate and matched training protocol before GPU work."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiments/force/f2_2_contact_gated_single_frame_load/index1'
OUT = BASE / 'result'
PARENT = ROOT / 'experiments/force/f2_1_load_temporal_encoding_ablation/index1/result'


def gate(position, threshold):
    """SO101 normalized position: lower is more closed; observation at t only."""
    return (np.asarray(position) <= threshold).astype(np.float32)


def main():
    if (OUT / 'gate_config.json').exists():
        raise RuntimeError('Gate already frozen; no silent refreezing')
    split = json.loads((PARENT / 'split_manifest.json').read_text(encoding='utf-8'))
    raw = Path(json.loads((PARENT / 'r1_preparation.json').read_text())['raw'])
    positions = []
    for ep in split['train']:
        with np.load(raw / f'ep{ep:03d}.npz') as z:
            positions.append(z['state'][:, 5].copy())
    values = np.concatenate(positions).astype(np.float64)
    assert len(values) == 17243 and np.isfinite(values).all()
    # Fixed calibration algorithm, not a search over policy performance.
    low, high = np.quantile(values, [.1, .9])
    threshold = float((low + high) / 2)
    assert high > low
    activity = float(gate(values, threshold).mean())
    OUT.mkdir(parents=True, exist_ok=True)

    def save(name, value):
        (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')

    save('gate_config.json', dict(style='G1', rule='float(raw gripper.pos[t] <= threshold)',
        threshold=threshold, calibration='midpoint of Train36 pooled gripper.pos q10 and q90',
        q10=float(low), q90=float(high), train_frames=len(values), train_active_fraction=activity,
        inputs=['raw gripper.pos[t]'], position_semantics='smaller means more closed',
        limitation='grasp-regime heuristic, NOT measured contact; may activate during empty closure',
        fit_episodes=split['train'], no_dev_calibration=True, no_load_or_command_input=True,
        degeneracy_thresholds=dict(always_off_max=.05, always_on_min=.95),
        encoded_feature_gating=True, token_and_positional_embedding_retained=True,
        frozen_utc=datetime.now(timezone.utc).isoformat()))
    protocol = dict(steps=100000, batch_size=4, seed=1, save_eval_every=20000,
        selection='lowest frame-weighted Dev12 normalized h0 gripper L1; earlier step breaks ties',
        initialization='fresh matched seed1; no trained A1 warmstart', optimizer='AdamW',
        lr=1e-5, backbone_lr=1e-4, weight_decay=1e-4, gradient_clip=10,
        nonfinite='fail and preserve evidence; no automatic retry', amp=False,
        chunk_size=100, final_sealed=True, parent='F2.1-r1 A1 architecture, not historical 8D P1',
        contract_sha256=hashlib.sha256((BASE / 'decision_F2_2_CONTACT_GATED_SINGLE_FRAME_LOAD.md').read_bytes()).hexdigest(),
        gate_sha256=hashlib.sha256((OUT / 'gate_config.json').read_bytes()).hexdigest())
    save('protocol.json', protocol)
    (OUT / 'variant_configs').mkdir(exist_ok=True)
    for variant in ['B0', 'B1']:
        save(f'variant_configs/{variant}.json', dict(**protocol, variant=variant,
            load_history_steps=1, load_feature_gating=variant == 'B1'))
    save('preparation.json', dict(status='FROZEN_PENDING_GPU_GATES', train_activity=activity,
        training_launched=False, final_sample_reads=0))
    print(json.dumps(dict(threshold=threshold, q10=float(low), q90=float(high), activity=activity)))


if __name__ == '__main__':
    main()
