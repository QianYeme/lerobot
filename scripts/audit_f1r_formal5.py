"""Audit original Formal5 collection and command equivalence before load-effect analysis."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/force/f1r_actuator_load_contact_intervention/index1/result'


def main():
    run = ROOT / 'results/experiments/F1R_ACTUATOR_LOAD_CONTACT_INTERVENTION' / ('audit_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    run.mkdir(parents=True)
    records, signals, hashes = [], {}, {}
    for kind in ['formal5_kind1', 'formal5_kind2']:
        folder = ROOT / '数据集/formal5' / kind
        info = json.loads((folder / 'meta/info.json').read_text(encoding='utf-8'))
        frame = pd.read_parquet(folder / 'data/chunk-000/file-000.parquet')
        state_names = info['features']['observation.state']['names']
        action_names = info['features']['action']['names']
        gi, ci = state_names.index('gripper.pos'), action_names.index('gripper.pos')
        assert 'gripper.load' in state_names and ('gripper.curr' in state_names or 'gripper.current' in state_names)
        assert sorted(frame.episode_index.unique()) == list(range(5))
        hashes[kind] = {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in folder.rglob('*') if p.is_file() and '.cache' not in p.parts}
        for ep in range(5):
            z = frame[frame.episode_index == ep].sort_values('frame_index')
            state, action, timestamp = np.stack(z['observation.state']), np.stack(z.action), z.timestamp.to_numpy()
            assert np.array_equal(z.frame_index, np.arange(len(z)))
            assert np.isfinite(state).all() and np.isfinite(action).all() and np.isfinite(timestamp).all()
            cmd = action[:, ci]
            key = f'{kind}_ep{ep:03d}'
            signals[key] = {'state': state, 'action': action, 'command': cmd, 'timestamp': timestamp}
            np.savez_compressed(run / f'{key}.npz', state=state, action=action, timestamp=timestamp, frames=z.frame_index.to_numpy())
            records.append({'dataset': kind, 'episode': ep, 'condition': 'UNASSIGNED', 'frames': len(z),
                            'nominal_fps': info['fps'], 'timestamp_last_s': float(timestamp[-1]),
                            'timestamp_step_max_error_s': float(np.max(np.abs(np.diff(timestamp) - 1 / info['fps']))),
                            'state_dim': state.shape[1], 'state_names': state_names,
                            'command_first': float(cmd[0]), 'command_last': float(cmd[-1]),
                            'command_min': float(cmd.min()), 'command_max': float(cmd.max()),
                            'command_unique_values': int(len(np.unique(cmd))),
                            'command_changes': int(np.count_nonzero(np.diff(cmd))),
                            'arm_recorded_target_peak_to_peak': np.ptp(action[:, :5], axis=0).tolist(),
                            'arm_observed_position_peak_to_peak': np.ptp(state[:, :5], axis=0).tolist(),
                            'gripper_observed_position_min_max': [float(state[:, gi].min()), float(state[:, gi].max())],
                            'finite': True, 'trial_validity': 'UNVERIFIED_PENDING_VIDEO_SAFETY_PROTOCOL',
                            'raw': str(run / f'{key}.npz')})
    reference = signals['formal5_kind1_ep000']['command']
    comparisons = []
    for key, d in signals.items():
        same_length = len(reference) == len(d['command'])
        comparisons.append({'trial': key, 'reference': 'formal5_kind1_ep000',
                            'same_frame_count': same_length,
                            'command_bitwise_equal': same_length and bool(np.array_equal(reference, d['command'])),
                            'command_max_abs_difference': float(np.max(np.abs(reference - d['command']))) if same_length else None})
    result = {'raw': str(run), 'trials': records, 'command_equivalence': comparisons,
              'all_commands_bitwise_equal': all(r['command_bitwise_equal'] for r in comparisons),
              'load_current_effect_metrics_computed': False,
              'condition_mapping': 'PENDING_USER_CONFIRMATION',
              'protocol_freeze_before_collection': 'NOT_VERIFIED', 'video_review': 'NOT_RUN'}
    (OUT / 'collection_preflight.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    (run / 'data_sha256.json').write_text(json.dumps(hashes, ensure_ascii=False, indent=2), encoding='utf-8')
    (run / 'collection_preflight.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    figure_dir = OUT / 'figures'
    figure_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, kind in zip(axes, ['formal5_kind1', 'formal5_kind2']):
        for ep in range(5):
            d = signals[f'{kind}_ep{ep:03d}']
            ax.plot(d['timestamp'], d['command'], label=f'ep{ep}')
        ax.set(title=kind + ' recorded targets', xlabel='Dataset timestamp (s)', ylabel='Gripper target (native units)')
        ax.legend()
    fig.tight_layout(); fig.savefig(figure_dir / 'recorded_command_equivalence.png', dpi=150); plt.close(fig)
    print(json.dumps({'trials': len(records), 'all_commands_bitwise_equal': result['all_commands_bitwise_equal'],
        'command_equivalence': comparisons,
        'trial_summary': [{k: r[k] for k in ['dataset', 'episode', 'frames', 'command_first', 'command_last', 'command_changes', 'arm_observed_position_peak_to_peak']} for r in records],
        'raw': str(run)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
