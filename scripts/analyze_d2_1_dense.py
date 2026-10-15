"""Analyze frozen P1 cache against reviewed intervals, without physical hold proxies."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/position/d2_1_evidence_completion/index1/result'
RAW = ROOT / 'results/experiments/D2_1_ACTION_CHAIN_EVIDENCE_COMPLETION'

# Review windows and released-by upper bounds, NOT exact release-onset brackets.
DEV = {
    3: ((382, 410), (260, 290)), 5: ((368, 392), (240, 270)),
    12: ((395, 426), (248, 286)), 15: ((392, 422), (242, 278)),
    21: ((390, 440), (255, 290)), 28: ((400, 450), (275, 300)),
    33: ((400, 450), (280, 305)), 39: ((400, 450), (280, 310)),
    41: ((416, 440), (270, 292)), 47: ((368, 396), (250, 280)),
    53: ((386, 414), (265, 294)), 58: ((410, 440), (282, 310)),
}


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def ensemble(chunks):
    result = []
    for t in range(len(chunks)):
        origins = np.arange(max(0, t - chunks.shape[1] + 1), t + 1)
        weights = np.exp(-0.01 * np.arange(len(origins)))
        result.append(np.average(chunks[origins, t - origins], axis=0, weights=weights))
    return np.asarray(result)


def measurements(sent, unclipped, h0, target, state, mask):
    if not mask.any():
        return None
    error = np.abs(sent[mask] - target[mask]).mean(axis=0)
    return {
        'frames': int(mask.sum()), 'per_joint_mae': error.tolist(),
        'arm_mae': float(error[:5].mean()), 'gripper_mae': float(error[5]),
        'per_joint_action_minus_state_mean': (sent[mask] - state[mask, :6]).mean(axis=0).tolist(),
        'per_joint_action_minus_state_abs_mean': np.abs(sent[mask] - state[mask, :6]).mean(axis=0).tolist(),
        'per_joint_clip_rate': (np.abs(sent[mask] - unclipped[mask]) > 1e-6).mean(axis=0).tolist(),
        'per_joint_processing_delta': np.abs(sent[mask] - h0[mask]).mean(axis=0).tolist(),
    }


def main():
    results, hashes, rows = {}, {}, []
    reference = json.loads((ROOT / 'experiments/position/d2_action_chain_diagnosis/index1/result/p1_unphased_metrics.json').read_text())
    checks = []
    for ep, (release, plateau) in DEV.items():
        path = RAW / 'P1_cache' / f'episode_{ep:03d}.npz'
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as cache:
            frame, state, target = cache['frames'], cache['state'], cache['target']
            chunks, normalized = cache['chunks'], cache['normalized_chunks']
        assert np.array_equal(frame, np.arange(len(frame)))
        assert chunks.shape == normalized.shape == (len(frame), 100, 6)
        assert state.shape == (len(frame), 8) and target.shape == (len(frame), 6)
        assert all(np.isfinite(a).all() for a in (state, target, chunks, normalized))
        h0, temporal = chunks[:, 0], ensemble(chunks)
        masks = {
            'full_task': np.ones(len(frame), dtype=bool),
            'released_retreat_core': frame > release[1],
            'release_review_window_NOT_EVENT_TIMING': (frame >= release[0]) & (frame <= release[1]),
            'elevated_posture_candidate_NOT_AIRBORNE_HOLD': (frame >= plateau[0]) & (frame <= plateau[1]),
        }
        covered = masks['released_retreat_core']
        masks['unknown_physical_phase'] = ~covered
        results[str(ep)] = {}
        for config, pred, limit in [('raw_h0', h0, None), ('off+5', h0, 5), ('off+2', h0, 2),
                                    ('0.01+5', temporal, 5), ('0.01+2', temporal, 2)]:
            sent = pred if limit is None else state[:, :6] + np.clip(pred - state[:, :6], -limit, limit)
            results[str(ep)][config] = {}
            for phase, mask in masks.items():
                stats = measurements(sent, pred, h0, target, state, mask)
                results[str(ep)][config][phase] = stats
                rows.append({'model': 'P1', 'episode': ep, 'config': config, 'phase': phase,
                             'status': 'VISUAL_CORE' if phase == 'released_retreat_core' else
                             ('UNPHASED' if phase in ('full_task', 'unknown_physical_phase') else 'UNCERTAIN_AUXILIARY_NOT_PRIMARY'),
                             **stats})
            old = reference['episodes'][str(ep)]
            expected = old['h0_per_joint_mae'] if limit is None else old['replay'][f'{None if config.startswith("off") else 0.01}_{limit}']['per_joint_mae']
            checks.append(float(np.max(np.abs(np.asarray(expected) - results[str(ep)][config]['full_task']['per_joint_mae']))))
    assert max(checks) < 1e-3, max(checks)
    aggregate = {}
    for config in results['3']:
        aggregate[config] = {}
        for phase in results['3'][config]:
            values = [episode[config][phase] for episode in results.values()]
            count = sum(x['frames'] for x in values)
            aggregate[config][phase] = {'frames': count}
            for key in values[0]:
                if key != 'frames':
                    aggregate[config][phase][key] = np.average([x[key] for x in values], weights=[x['frames'] for x in values], axis=0).tolist()
    save('dense_phase_metrics.json', {'teacher_forced': True, 'physical_units': 'dataset native joint units; not mm or force',
         'primary_reliable_core': 'released_retreat_core only; not release onset or full release phase',
         'hold_evaluated': False, 'episodes': results, 'frame_weighted_aggregate': aggregate})
    with (OUT / 'per_episode_phase_metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    save('dense_cache_validation.json', {'status': 'PASS', 'episodes': 12, 'frames': 5748,
         'shape_and_finite_and_contiguous_frame_checks': 'PASS', 'existing_replay_mae_max_abs_difference': max(checks),
         'cache_sha256': hashes, 'temporal_weights': 'oldest origin first, exp(-0.01*i); reset every episode',
         'clamp_state': 'recorded expert state, NOT fresh motor state', 'normalized_path': 'previous replay_equivalence_validation.json retained'})
    print(json.dumps({'episodes': len(results), 'frames': 5748, 'replay_check': max(checks)}))


if __name__ == '__main__':
    main()
