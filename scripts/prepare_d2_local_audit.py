"""D2 metadata-only preflight. Never opens episode data, labels or videos."""
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/position/d2_action_chain_diagnosis/index1/result'
RAW = ROOT / 'results/experiments/D2_P1_ACTION_CHAIN_DIAGNOSIS'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    split_path = ROOT / 'experiments/data/d1_training_readiness/index1/result/split_manifest_v1.json'
    info_path = ROOT / '数据集/formal4/kind_merged_nomaster/meta/info.json'
    split = json.loads(split_path.read_text(encoding='utf-8'))
    info = json.loads(info_path.read_text(encoding='utf-8'))
    groups = {k: [e['episode_index'] for e in split['episodes'] if e['split'] == k]
              for k in ('train', 'dev', 'final_test')}
    assert len(groups['train']) == 36 and len(groups['dev']) == 12
    assert not set(groups['train']) & set(groups['dev'])
    assert not (set(groups['train']) | set(groups['dev'])) & set(groups['final_test'])
    files = [split_path, info_path, ROOT / 'src/lerobot/scripts/control_act_det.py',
             ROOT / 'src/lerobot/scripts/lerobot_record.py',
             ROOT / 'src/lerobot/policies/act/modeling_act.py',
             ROOT / 'src/lerobot/robots/so_follower/so_follower.py',
             ROOT / 'src/lerobot/robots/utils.py',
             ROOT / 'experiments/position/d2_action_chain_diagnosis/index1/decision.md']
    provenance = {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  'git_status': subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
                  'file_sha256': {str(p.relative_to(ROOT)): digest(p) for p in files},
                  'dependencies': {k: bool(importlib.util.find_spec(k)) for k in ('torch', 'pyarrow', 'pandas')},
                  'train_episodes': groups['train'], 'dev_episodes': groups['dev'],
                  'final_test_episode_contents_read': False,
                  'checkpoint_processor_prototype_hashes': None}
    schema = {'scope': 'metadata_only', 'state': info['features']['observation.state'],
              'action': info['features']['action'], 'fps_declared': info['fps'],
              'metadata_dimensions_match': info['features']['observation.state']['shape'] == [8]
              and info['features']['action']['shape'] == [6],
              'units_alignment_actual_feedback_and_reachability': 'UNVERIFIED',
              'full_schema_gate': 'INCOMPLETE'}
    deployment = {'evidence': 'current repository code inspection',
                  'temporal_weight': 'exp(-coeff * index); index 0 is oldest prediction',
                  'positive_coeff_favors': 'older predictions',
                  'ensemble_enabled': 'predict full chunk each step; bypass action queue',
                  'ensemble_off': 'queue consumes n_action_steps before next prediction',
                  'relative_limit': 'applied in follower.send_action against fresh Present_Position',
                  'recording_action': 'lerobot_record stores sent_action after follower limit',
                  'historical_recording_code_identity': 'UNVERIFIED',
                  'reported_runtime_config': {'temporal_ensemble_coeff': 0.01, 'max_relative_target': 2},
                  'actual_runtime_config': 'UNVERIFIED_NO_LOGS',
                  'four_configuration_replay': 'BLOCKED_CHECKPOINTS_AND_TORCH',
                  'real_robot_commands_sent': False}
    for name, value in [('provenance.json', provenance), ('schema_audit.json', schema),
                        ('deployment_contract_audit.json', deployment)]:
        (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    (RAW / 'local_preflight.json').write_text(json.dumps({'metadata_only': True,
        'final_test_read': False, 'training': False, 'robot': False,
        'result_files': ['provenance.json', 'schema_audit.json', 'deployment_contract_audit.json']}, indent=2), encoding='utf-8')
    print('D2 metadata audit saved; prediction and event audits remain incomplete.')


if __name__ == '__main__':
    main()
