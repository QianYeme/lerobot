"""Read-only scientific-input preflight; never launch F2.1 training on ambiguous baselines."""
import hashlib
import json
import subprocess
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiments/force/f2_1_load_temporal_encoding_ablation/index1'
OUT = BASE / 'result'
REPO = 'QYyyyyyyy/G1_P1_OCACT_100k'
REVISION = '33a605feb8c29dab284c3aa38740df11d2ae7835'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    evidence = OUT / 'baseline_metadata'
    evidence.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.mount('https://', HTTPAdapter(max_retries=Retry(total=2, backoff_factor=.5)))
    downloaded, errors = {}, {}
    for name in ['config.json', 'train_config.json', 'policy_preprocessor.json', 'policy_postprocessor.json']:
        path = evidence / name
        url = f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}'
        try:
            response = session.get(url, timeout=25)
            response.raise_for_status()
            value = response.json()
            if path.exists():
                assert json.loads(path.read_text(encoding='utf-8')) == value, 'Pinned metadata changed'
            else:
                path.write_bytes(response.content)
            downloaded[name] = dict(url=url, sha256=sha(path))
        except (requests.RequestException, ValueError) as error:
            errors[name] = type(error).__name__
    dataset_path = ROOT / '数据集/formal4/kind_merged_nomaster/meta/info.json'
    split_path = ROOT / 'experiments/data/d1_training_readiness/index1/result/split_manifest_v1.json'
    names = json.loads(dataset_path.read_text(encoding='utf-8'))['features']['observation.state']['names']
    split = json.loads(split_path.read_text(encoding='utf-8'))
    train = json.loads((ROOT / 'experiments/position/t0_discovery_act_vs_ocact/index1/result/training_manifest.json').read_text(encoding='utf-8'))
    config_path = evidence / 'config.json'
    config = json.loads(config_path.read_text(encoding='utf-8')) if config_path.exists() else None
    training_path = evidence / 'train_config.json'
    training = json.loads(training_path.read_text(encoding='utf-8')) if training_path.exists() else None
    counts = {part:sum(r['split']==part for r in split['episodes']) for part in ['train','dev','final_test']}
    assert counts == {'train':36,'dev':12,'final_test':12}
    if training:
        assert training['dataset']['episodes'] == train['train_episodes']
    identity = dict(hf_repo=REPO, hf_revision=REVISION, metadata_files=downloaded, fetch_errors=errors,
        local_git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        checkpoint_reference='results/experiments/T0_POSITION_DISCOVERY_ACT_VS_OCACT/P1_ocfix2_s1/checkpoints/100000/pretrained_model',
        dataset_state_names=names, dataset_metadata_sha256=sha(dataset_path),
        split_sha256=sha(split_path), split_counts=counts, final_sample_contents_read=False,
        checkpoint_config=config, baseline_training_config=training,
        verified_scope='pinned published configuration + original dataset metadata + source paths; NOT strict weight reload',
        strict_weights_forward_verified=False, policy_code_modified=False,
        source_paths=['src/lerobot/policies/act/modeling_act.py','src/lerobot/policies/act/processor_act.py'])
    (OUT / 'baseline_identity.json').write_text(json.dumps(identity,ensure_ascii=False,indent=2),encoding='utf-8')
    metrics = dict(OVERALL_STATUS='BLOCKED', F2_1_COMPLETE=False, SCIENTIFIC_OUTCOME='UNRESOLVED',
        blocker='BLOCKED_INPUT_SEMANTICS',
        reasons=['Existing P1 uses 8D state including both gripper.load and gripper.curr',
                 'Preserving P1 exactly does not yield a load-free/current-free baseline',
                 'Removing/masking those inputs changes baseline semantics and is not silently authorized'],
        baseline_identity_gate='CONFIG_VERIFIED_WEIGHTS_FORWARD_NOT_VERIFIED',
        secondary_primary_metric_note='Existing G0 normalized h0 gripper L1 and D2 physical h0 gripper MAE exist; choose and freeze exact existing protocol after semantics clarification',
        training_started=False, variants_implemented=False, final_test_read=False)
    (OUT / 'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(json.dumps(dict(metadata_downloaded=list(downloaded),errors=errors,state_fields=names,
                         split_counts=counts,status='BLOCKED_INPUT_SEMANTICS'),ensure_ascii=False))


if __name__ == '__main__':
    main()
