"""Manual reset records initial evidence without a Home gate."""
from types import SimpleNamespace
import pytest
from lerobot.utils import r5_10_protocol as p


def test_manual_reset_needs_no_home_file():
    args=SimpleNamespace(robot_id='nn',model='M1',position='A')
    obs={k:35. for k in p.POSITION_FIELDS};obs['gripper.load']=42.
    report=p.audit_initial(obs,args,{})
    assert report['status']=='RECORDED_MANUAL_RESET' and report['valid'] is None
    assert report['state']=={k:35. for k in p.POSITION_FIELDS}
    assert report['load']==42. and report['gripper_position']==35.
    assert 'home_reference' not in report and 'within_tolerance' not in report


def test_nonfinite_sensor_values_still_rejected():
    args=SimpleNamespace(robot_id='nn',model='M1',position='A')
    obs={k:0. for k in p.POSITION_FIELDS};obs['gripper.load']=float('nan')
    with pytest.raises(ValueError,match='Nonfinite'):p.audit_initial(obs,args,{})


def test_cli_and_four_commands_have_no_home_requirement():
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]
    source=(root/'src/lerobot/scripts/control_r5_10.py').read_text(encoding='utf-8')
    assert '--home-reference' not in source and 'load_home_reference' not in source
    guide=(root/'experiments/force/r5_10_real_robot_transfer_validation/index1/REAL_ROBOT_TEST_GUIDE.md').read_text(encoding='utf-8')
    assert '--home-reference' not in guide
    for model in p.MODELS:assert '--execute --model '+model in guide
    assert '--policy.temporal_ensemble_coeff=0.01' in guide
    assert '--robot.max_relative_target=5' in guide


def test_exact_primary_model_order_and_stage_semantics():
    assert p.SCHEDULE[:9]==[(position,model) for position in ['A','B','C'] for model in ['M1','M2','M3']]
    assert p.EXPECTED['M1']==('clip10',3,[80000])
    assert p.EXPECTED['M3']==('clip5',2,[60000,80000])
    assert p.STAGE_NAMES['P3']=='physical grasp established'
    assert p.STAGE_NAMES['P4']=='cup lifted'


def test_manual_reset_loop_runs_and_keeps_initial_evidence(tmp_path,monkeypatch):
    import importlib.util
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]
    spec=importlib.util.spec_from_file_location('r510_mock_io',root/'tests/scripts/test_f5_control.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    original=module.control.run_trial
    def route(args,runtime,common):
        args.model='M1'
        return original(args,runtime,common,protocol=p)
    monkeypatch.setattr(module.control,'run_trial',route)
    monkeypatch.setattr(module.protocol,'trial_id',p.trial_id)
    module.test_mock_trial_assistance_or_ctrl_c_cleanup_no_stop_lock(tmp_path,monkeypatch,False)
    import json
    report=json.loads((tmp_path/'session'/p.trial_id('M1','A')/'initial_state.json').read_text())
    assert report['status']=='RECORDED_MANUAL_RESET' and report['valid'] is None
