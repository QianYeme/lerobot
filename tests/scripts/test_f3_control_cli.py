import importlib.util
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT=Path(__file__).parents[2]
sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('control_f3',ROOT/'src/lerobot/scripts/control_f3.py')
control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control)


def test_parser_accepts_legacy_robot_camera_flags():
    args=control.parser().parse_args(['--bundle','model','--session-dir','session','--dry-run',
        '--robot.port=/dev/ttyACM0','--robot.id=nn','--policy.n_action_steps=1','--dataset.fps=30'])
    assert args.robot_port=='/dev/ttyACM0' and args.robot_id=='nn'
    assert args.policy_n_action_steps==1 and args.dataset_fps==30 and not args.execute


def test_prepare_is_unapproved_and_execute_stops_before_model_loading(tmp_path,monkeypatch):
    bundle=tmp_path/'bundle';bundle.mkdir();(bundle/'model_identity.json').write_text('{}')
    names=['normalization.json','local_normalization.json','assets/cup_prototype_bank_v1.pt',
        'global/config.json','models/M0/config.json','models/M2/config.json','local/010000.pt',
        'models/M0/policy_preprocessor.json','models/M0/policy_postprocessor.json',
        'models/M0/policy_preprocessor_step_3_normalizer_processor.safetensors',
        'models/M0/policy_postprocessor_step_0_unnormalizer_processor.safetensors']
    for name in names:
        path=bundle/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'fixture')
    manifest={path.relative_to(bundle).as_posix():hashlib.sha256(path.read_bytes()).hexdigest() for path in bundle.rglob('*') if path.is_file()}
    (bundle/'manifest.json').write_text(json.dumps(dict(sha256=manifest)))
    calib=tmp_path/'calibration';calib.mkdir();(calib/'nn.json').write_text('{}')
    session=tmp_path/'session'
    common=['control_f3','--bundle',str(bundle),'--session-dir',str(session),'--robot.calibration_dir',str(calib)]
    monkeypatch.setattr(sys,'argv',common+['--prepare-only']);control.main()
    review=json.loads((session/'stage0_review.json').read_text())
    assert review['passed'] is False
    monkeypatch.setattr(sys,'argv',common+['--execute','--model','M0','--trial-id','R01_M0','--stage0-report',str(session/'stage0_review.json')])
    with pytest.raises(ValueError,match='Human review'):
        control.main()
