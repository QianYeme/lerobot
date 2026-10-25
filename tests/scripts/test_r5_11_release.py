"""R5.11 direct commands and historical-only133k release boundaries."""
import importlib.util
import json
import re
import shlex
from pathlib import Path
import pytest
from lerobot.scripts.control_f5 import parser,settings
from lerobot.utils import r5_11_protocol as p

ROOT=Path(__file__).resolve().parents[2]


def test_four_guide_commands_are_valid_and_cli_limits_editable():
    guide=(ROOT/'experiments/force/r5_11_real_robot_133k_breakthrough_diagnostic/index1/REAL_ROBOT_TEST_GUIDE.md').read_text(encoding='utf-8')
    commands=re.findall(r'python src/lerobot/scripts/control_r5_11\.py (.*?)(?=\n```)',guide,re.S)
    commands=[c for c in commands if '--execute' in c];assert len(commands)==4
    models=[]
    for cmd in commands:
        cmd=cmd.replace('\\\n',' ').split('2>&1')[0].replace('--policy.temporal_ensemble_coeff=0.01','--policy.temporal_ensemble_coeff=0.025').replace('--robot.max_relative_target=5','--robot.max_relative_target=2')
        cli=parser(p.MODELS);cli.add_argument('--trial-kind');args=cli.parse_args(shlex.split(cmd))
        common=settings(args);assert common['max_relative_target']==2 and common['temporal_ensemble_coeff']==.025
        assert args.position=='A' and args.trial_kind=='stage_a';models.append(args.model)
    assert models==p.MODELS and '--home-reference' not in guide


def test_r3_bundle_rejects_other_models(tmp_path):
    with pytest.raises(ValueError,match='only R3'):p.verify_bundle(tmp_path,'R2')
    (tmp_path/'model_identity.json').write_text(json.dumps({'R3':{},'R2':{}}))
    with pytest.raises(ValueError,match='Only the historical'):p.verify_bundle(tmp_path)


def test_stage_order_and_historical133k_identity():
    assert p.SCHEDULE[:4]==[('A',m) for m in p.MODELS]
    assert p.R3_SHA=='d094de659f58df3e55cd6284a83729cfd8658fb1d4b19f73d6a8cb634a61fc0f'
    assert p.STAGE_NAMES['P3']=='physical grasp established'


def test_r511_loop_records_reference_schema_without_home(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('r511_io',ROOT/'tests/scripts/test_f5_control.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    original=module.control.run_trial
    def route(args,runtime,common):
        args.model='R0';runtime.identity['state_fields']=p.POSITION_FIELDS+['gripper.load','gripper.curr']
        return original(args,runtime,common,protocol=p)
    monkeypatch.setattr(module.control,'run_trial',route)
    monkeypatch.setattr(module.protocol,'trial_id',p.trial_id)
    module.test_mock_trial_assistance_or_ctrl_c_cleanup_no_stop_lock(tmp_path,monkeypatch,False)
    run=tmp_path/'session'/p.trial_id('R0','A')
    assert len(json.loads((run/'configuration.json').read_text())['state_order'])==8
    assert json.loads((run/'initial_state.json').read_text())['status']=='RECORDED_MANUAL_RESET'
