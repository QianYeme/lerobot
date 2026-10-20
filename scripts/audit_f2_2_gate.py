"""Read only frozen Train/Dev caches; descriptive commands never enter the gate."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/force/f2_2_contact_gated_single_frame_load/index1/result'
PARENT = ROOT / 'experiments/force/f2_1_load_temporal_encoding_ablation/index1/result'


def main():
    spec = json.loads((OUT / 'gate_config.json').read_text())
    split = json.loads((PARENT / 'split_manifest.json').read_text())
    raw = Path(json.loads((PARENT / 'r1_preparation.json').read_text())['raw'])
    rows, bins = [], []
    for role in ['train', 'dev']:
        for ep in split[role]:
            with np.load(raw / f'ep{ep:03d}.npz') as z:
                pos, command = z['state'][:, 5], z['action'][:, 5]
                active = pos <= spec['threshold']
                rows.append(dict(split=role, episode=ep, frames=len(pos), active=int(active.sum()),
                    fraction=float(active.mean()), pos_min=float(pos.min()), pos_max=float(pos.max())))
                for variable, values in [('position', pos), ('recorded_command_descriptive_only', command)]:
                    for low in range(-20, 120, 10):
                        subset = (values >= low) & (values < low + 10)
                        bins.append(dict(split=role, episode=ep, variable=variable, low=low, high=low+10,
                            frames=int(subset.sum()), active=int(active[subset].sum())))
    for name, records in [('gate_activity.csv', rows), ('gate_activity_bins.csv', bins)]:
        with (OUT / name).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    result = dict(gate_sha256=hashlib.sha256((OUT/'gate_config.json').read_bytes()).hexdigest(),
                  frozen_before_dev_audit=True, final_samples_read=0, activity={})
    for role in ['train', 'dev']:
        records = [r for r in rows if r['split'] == role]
        fraction = sum(r['active'] for r in records) / sum(r['frames'] for r in records)
        result['activity'][role] = dict(fraction=fraction, global_degenerate=fraction <= .05 or fraction >= .95,
            degenerate_episodes=[r['episode'] for r in records if r['fraction'] <= .05 or r['fraction'] >= .95])
    (OUT/'gate_activity_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
