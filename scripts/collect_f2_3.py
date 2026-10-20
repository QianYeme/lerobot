"""Verify/extract downloaded F2.3 package and independently audit cached predictions."""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/force/f2_3_global_vs_local_load_regulation/index1'


def main():
    package=BASE/'F2_3_result.zip'
    expected='4fa92338b4a2d5053e816aa360aaecb9659aeaf3d4a4b089667a827a39c84b10'
    assert hashlib.sha256(package.read_bytes()).hexdigest()==expected
    with zipfile.ZipFile(package) as archive:
        assert archive.testzip() is None
        for member in archive.infolist():
            assert (BASE/member.filename).resolve().is_relative_to(BASE.resolve())
            assert member.filename.startswith('result/')
        if not (BASE/'pretraining_archive').exists():shutil.copytree(BASE/'result',BASE/'pretraining_archive')
        archive.extractall(BASE)
    out=BASE/'result'
    metrics=json.loads((out/'metrics.json').read_text())
    std=json.loads((ROOT/'experiments/force/f2_1_load_temporal_encoding_ablation/index1/result/normalization.json').read_text())['action']['std']
    audit={}
    predictions={}
    for name in ['C1-no-load','C1-load']:
        with np.load(out/name/'selected_dev.npz') as z:
            prediction,target=z['prediction'],z['target']
            assert len(prediction)==5748
            error=np.abs(prediction-target)
            score=float(error[:,5].mean())
            assert abs(score-metrics['variants'][name]['primary'])<1e-8
            assert np.array_equal(prediction[:,:5],z['nominal'][:,:5])
            assert np.max(np.abs(z['delta']))<=4.832699459791184+1e-6
            expected_delta=(prediction[:,5]-z['nominal'][:,5])*std[5]
            assert np.max(np.abs(expected_delta-z['delta']))<1e-4
            predictions[name]=prediction.copy()
            audit[name]=dict(primary=score,arm_identical_to_nominal=True,residual_sign_and_bound=True)
    assert np.array_equal(predictions['C1-no-load'][:,:5],predictions['C1-load'][:,:5])
    score=metrics['variants']
    audit['primary_improvement_vs_C0']=1-score['C1-load']['primary']/score['C0']['primary']
    audit['primary_improvement_vs_local_no_load']=1-score['C1-load']['primary']/score['C1-no-load']['primary']
    audit['arm_error_ratio_vs_C0']=score['C1-load']['physical_arm_mae']/score['C0']['physical_arm_mae']
    (BASE/'local_collection_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
