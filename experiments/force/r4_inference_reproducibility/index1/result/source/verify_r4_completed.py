"""Read-only CPU checks of completed R4 raw output; writes a new audit only."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

WORK=Path('/root/autodl-tmp/R4_20261023')
OUT=WORK/'result'


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    observations=[]
    for seed in [3,1]:
        prefix=WORK/'raw/benchmark_off'/f'S{seed}'
        assert (prefix/'D0/input_manifest.json').read_bytes()==(prefix/'D1/input_manifest.json').read_bytes()
        first=np.load(prefix/'D0/replay.npz');second=np.load(prefix/'D1/replay.npz')
        for key in first.files:np.testing.assert_array_equal(first[key],second[key])
        assert len(first['episode'])==28739 and set(first['episode'])==set(range(60))
        pairs=list(zip(first['episode'].tolist(),first['frame'].tolist()))
        assert len(set(pairs))==28739
        files=sorted((prefix/'D0/chunks').glob('*.npz'))
        assert len(files)==7185==len(list((prefix/'D1/chunks').glob('*.npz')))
        h0=[];maximum=0.;samples=0
        for path in files:
            with np.load(path) as a,np.load(prefix/'D1/chunks'/path.name) as b:
                assert a['native'].shape[1:]==(100,6)
                np.testing.assert_array_equal(a['native'],b['native'])
                assert np.isfinite(a['native']).all()
                maximum=max(maximum,float(np.abs(a['native']-b['native']).max()))
                h0.append(a['native'][:,0]);samples+=len(a['native'])
        np.testing.assert_array_equal(np.concatenate(h0),first['prediction_native'])
        assert samples==28739
        observations.append(dict(seed=seed,batches=len(files),samples=samples,max_native=maximum,
            full_chunk_array_equal=True,cached_h0_matches_chunks=True,all_inputs_equal=True))
        first.close();second.close()
    with (OUT/'postfix_f5_1_metrics.csv').open() as stream:
        metrics={r['model']:r for r in csv.DictReader(stream)}
    for seed in [36,1,2,3,4]:
        label='G36-S3' if seed==36 else f'G60-S{seed}'
        tag='D0' if seed in [1,3] else 'POSTFIX'
        with np.load(WORK/'raw/benchmark_off'/f'S{seed}'/tag/'replay.npz') as data:
            delta=np.abs(data['prediction_native']-data['target_native'])
            assert np.isclose(float(delta[:,5].mean()),float(metrics[label]['gripper_mae_native']),rtol=1e-7,atol=1e-7)
            assert np.isclose(float(delta[:,:5].mean()),float(metrics[label]['arm_mae_native']),rtol=1e-7,atol=1e-7)
    config=json.loads((OUT/'audit_config.json').read_text())
    for path,expected in config['checkpoint_files'].items():assert digest(Path(path))==expected
    with (WORK.parent/'F5_1_r3_20261022/result/training_manifest.csv').open() as stream:
        original={r['model']:r for r in csv.DictReader(stream)}
    identities=json.loads((OUT/'postfix_f5_1/model_identity.json').read_text())
    for seed in [1,2,3,4]:assert identities[f'G60-S{seed}']['sha256']==original[f'G60-S{seed}']['sha256']
    for seed in [1,2,3]:assert identities[f'NewGL60-L{seed}']['local_sha256']==original[f'L60-S{seed}']['sha256']
    result=dict(status='PASS',method='Fresh CPU raw-array verification, no new model inference',
        repeated_full_chunks=observations,fresh_global_h0_metrics_verified=5,
        global_and_local_identity_cross_checks=7,checkpoint_files_rehashed=len(config['checkpoint_files']),
        limitations='No video re-decode or external review; local-output metrics not independently re-inferred')
    (WORK/'independent_cpu_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
