"""Verify immutable R4 delivery and package executor review; no model runs."""
import csv
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
INDEX=ROOT/'experiments/force/r4_inference_reproducibility/index1'
OUT=INDEX/'result'
CANON=ROOT/'results/experiments/R4_INFERENCE_REPRODUCIBILITY_AND_FINAL_DATASET_GATE/20261023_delivery'
RAW=OUT/'server_original/result'


def load(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def rows(name):
    with (RAW/name).open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def main():
    receipt=load(CANON/'delivery_receipt.json');archive_path=CANON/'R4_result.zip'
    assert archive_path.stat().st_size==receipt['bytes'] and digest(archive_path)==receipt['sha256']
    with zipfile.ZipFile(archive_path) as archive:assert archive.testzip() is None
    manifest=load(RAW/'manifest.json')['sha256']
    for name,expected in manifest.items():assert digest(RAW/name)==expected,name
    for path in (CANON/'extracted/result').rglob('*'):
        if path.is_file():assert digest(path)==digest(RAW/path.relative_to(CANON/'extracted/result'))
    decision=INDEX/'decision_R4_INFERENCE_REPRODUCIBILITY_AND_FINAL_DATASET_GATE.md'
    config=load(RAW/'audit_config.json')
    assert digest(decision)==config['decision_sha256']
    assert digest(RAW/'decision.md')==config['decision_sha256']
    for name,expected in config['scripts'].items():assert digest(ROOT/'scripts'/name)==expected,name
    required=['summary.md','validation.md','audit_config.json','frozen_input_manifest.json',
        'same_process_repeat.csv','cross_process_repeat.csv','preprocess_repeat.csv','full_replay_repeat.csv',
        'temporal_state_audit.md','determinism_config.md','seed1_repeat_verification.csv','postfix_f5_1_metrics.csv',
        'builder_review.md','red_team_review.md','proposal.md']
    assert all((OUT/name).is_file() for name in required)
    repeat=rows('full_replay_repeat.csv')
    assert {int(r['seed']) for r in repeat}=={1,3}
    for row in repeat:
        assert int(row['frames'])==28739 and int(row['batches'])==7185
        assert row['inputs_bitwise_equal']=='True' and row['passed']=='True'
        assert float(row['max_native'])==float(row['h0_max_native'])==0
    for name in ['same_process_repeat.csv','cross_process_repeat.csv','preprocess_repeat.csv']:
        values=rows(name)
        assert {int(r['seed']) for r in values if r['mode']=='benchmark_off'}=={1,3}
        assert all(r['passed']=='True' for r in values if r['mode']=='benchmark_off')
    model_rows={r['model']:r for r in rows('postfix_f5_1_metrics.csv')}
    assert len(model_rows)==9
    assert all(int(r['frames'])==28739 and 'NOT HELD-OUT' in r['scope'] for r in model_rows.values())
    for name in ['NewGL60-L1','NewGL60-L2','NewGL60-L3']:
        assert model_rows[name]['arm_mae_native']==model_rows['G60-S3']['arm_mae_native']
    original=ROOT/'results/experiments/F5_1_FULL60_MULTI_SEED_OPTIMAL/20261023T020523Z/result/model_identity.json'
    old=load(original);new=load(RAW/'postfix_f5_1/model_identity.json')
    for name in model_rows:
        key='sha256' if new[name]['kind']=='global' else 'local_sha256'
        assert new[name][key]==old[name][key],name
    local_values=np.array([float(model_rows[f'NewGL60-L{s}']['gripper_mae_native']) for s in [1,2,3]])
    local=rows('postfix_f5_1/local_seed_robustness.csv')[0]
    assert np.isclose(local_values.std(),float(local['std']),rtol=1e-12)
    cpu=load(OUT/'independent_cpu_validation.json')
    assert cpu['status']=='PASS' and cpu['checkpoint_files_rehashed']==12
    runtime=config['runtime']; baseline=runtime['historical/S3/B0/audit_config.json']
    control=runtime['benchmark_off/S3/B0/audit_config.json']
    assert baseline['cudnn_benchmark'] and not control['cudnn_benchmark']
    for key in ['amp','cudnn_deterministic','deterministic_algorithms','matmul_allow_tf32','cudnn_allow_tf32']:
        assert baseline[key]==control[key]
    authored=['README.md','summary.md','validation.md','metrics.json','proposal.md','builder_review.md','red_team_review.md']
    for name in authored:
        if not name.endswith('.md'):continue
        for target in re.findall(r'\]\(([^)]+)\)',(OUT/name).read_text(encoding='utf-8')):
            if '://' not in target:assert (OUT/target).exists(),target
    save(OUT/'delivery_validation.json',dict(status='PASS',original_zip_sha256=receipt['sha256'],
        original_zip_crc='PASS',original_manifest_members_verified=len(manifest),
        original_reports_byte_identical=True,contract_sha_verified=config['decision_sha256'],
        frozen_worker_scripts_verified=len(config['scripts']),required_contract_files=len(required),
        audited_seeds=[1,3],frames_each=28739,horizons=100,actions=6,
        recomputed_model_rows=9,identity_matches_historical_models=9,
        only_intended_runtime_flag_changed=True,server_cpu_validation=cpu,
        external_reviews='PENDING',R4_COMPLETE=False,R4_REPRODUCIBILITY_PASS=True))
    source=OUT/'source';source.mkdir(exist_ok=True)
    for name in ['run_r4_repro.py','queue_r4_repro.py','finalize_r4_repro.py','verify_r4_completed.py',
                 'package_r4_reviewed.py','launch_r4.sh','launch_r4_queue.sh']:
        shutil.copy2(ROOT/'scripts'/name,source/name)
    shutil.copy2(ROOT/'tests/scripts/test_r4_repro_contract.py',source/'test_r4_repro_contract.py')
    selected=[OUT/name for name in authored+required if (OUT/name).exists()]
    selected += [OUT/'delivery_validation.json',OUT/'independent_cpu_validation.json']
    selected += [p for directory in [RAW,source] for p in directory.rglob('*') if p.is_file()]
    selected += [decision]
    members={str(path.relative_to(INDEX)).replace('\\','/'):path for path in selected}
    members['server_delivery_receipt.json']=CANON/'delivery_receipt.json'
    save(OUT/'delivery_manifest.json',dict(sha256={name:digest(path) for name,path in sorted(members.items())}))
    members['result/delivery_manifest.json']=OUT/'delivery_manifest.json'
    bundle_path=INDEX/'R4_REVIEWED_result_20261023.zip'
    with zipfile.ZipFile(bundle_path,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,path in sorted(members.items()):archive.write(path,name)
    with zipfile.ZipFile(bundle_path) as archive:
        assert archive.testzip() is None
        for name,expected in load(OUT/'delivery_manifest.json')['sha256'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest()==expected,name
    final=dict(path=str(bundle_path.relative_to(ROOT)),bytes=bundle_path.stat().st_size,sha256=digest(bundle_path),
        members=len(members),crc='PASS',member_sha256='PASS',status='READY_AWAIT_EXTERNAL_REVIEW')
    save(bundle_path.with_suffix('.receipt.json'),final)
    print(json.dumps(final,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
