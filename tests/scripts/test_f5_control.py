"""F5 CLI/metadata and mock-I/O regressions. No physical robot connection."""
import csv
import importlib.util
import json
import re
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from lerobot.utils import f5_protocol as protocol
spec=importlib.util.spec_from_file_location('control_f5',ROOT/'src/lerobot/scripts/control_f5.py')
control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control)


def args(tmp_path,extra=None):
    return control.parser().parse_args(['--bundle',str(tmp_path/'bundle'),'--session-dir',str(tmp_path/'session'),
        '--execute','--model','Strong-G','--position','A']+(extra or []))


def test_exact_twelve_trial_ids_and_blocked_rotation():
    assert len(protocol.SCHEDULE)==12 and len(set(protocol.SCHEDULE))==12
    assert [m for p,m in protocol.SCHEDULE if p=='B']==['Strong-G','L-Load','New-GL','P1-Hist']
    assert [m for p,m in protocol.SCHEDULE if p=='C']==['L-Load','New-GL','P1-Hist','Strong-G']
    assert protocol.trial_id('New-GL','C')=='F5_NewGL_C_01'


def test_cli_tunables_not_hardcoded_and_recorded(tmp_path):
    a=args(tmp_path,['--robot.max_relative_target=2','--policy.temporal_ensemble_coeff=0.025','--robot.id=other'])
    common=control.settings(a)
    assert common['max_relative_target']==2 and common['temporal_ensemble_coeff']==.025 and common['robot_id']=='other'
    default=control.settings(args(tmp_path));assert default['max_relative_target']==5 and default['temporal_ensemble_coeff']==.01
    for value in ['0','nan','inf','-1']:
        with pytest.raises(ValueError):control.settings(args(tmp_path,['--robot.max_relative_target='+value]))


def test_main_forwards_manual_ensemble_to_runtime(tmp_path,monkeypatch):
    captured=[]
    class Runtime:
        def __init__(self,*values):captured.extend(values)
        def step(self,obs):return np.zeros(6),np.zeros(6),None
        def reset(self):pass
    fake=SimpleNamespace(set_num_threads=lambda n:None,backends=SimpleNamespace(cudnn=SimpleNamespace(),cuda=SimpleNamespace(matmul=SimpleNamespace())))
    monkeypatch.setitem(sys.modules,'torch',fake)
    monkeypatch.setitem(sys.modules,'lerobot.policies.act.f5_runtime',SimpleNamespace(F5Runtime=Runtime))
    monkeypatch.setattr(sys,'argv',['control_f5','--bundle',str(tmp_path/'bundle'),'--session-dir',str(tmp_path/'session'),
        '--model','Strong-G','--position','A','--dry-run','--robot.max_relative_target=2','--policy.temporal_ensemble_coeff=.025'])
    control.main();assert captured[-2:]==[1,.025]


def test_guide_four_commands_accept_manual_tunables():
    guide=(ROOT/'experiments/force/f5_unified_real_robot_validation/index1/REAL_ROBOT_TEST_GUIDE.md').read_text(encoding='utf-8')
    commands=re.findall(r'python src/lerobot/scripts/control_f5\.py (.*?)(?=\n```)',guide,re.S)
    commands=[command for command in commands if '--execute' in command]
    assert len(commands)==4
    models=[]
    for command in commands:
        command=command.replace('\\\n',' ').split('2>&1')[0]
        command=re.sub(r'\$\{F5_POSITION[^}]*\}','A',command)
        command=command.replace('--policy.temporal_ensemble_coeff=0.01','--policy.temporal_ensemble_coeff=0.025')
        command=command.replace('--robot.max_relative_target=5','--robot.max_relative_target=2')
        parsed=control.parser().parse_args(shlex.split(command))
        common=control.settings(parsed)
        assert common['temporal_ensemble_coeff']==.025 and common['max_relative_target']==2
        models.append(parsed.model)
    assert models==list(protocol.MODELS)


@pytest.mark.parametrize('interrupt',[False,True])
def test_mock_trial_assistance_or_ctrl_c_cleanup_no_stop_lock(tmp_path,monkeypatch,interrupt):
    a=args(tmp_path,['--robot.max_relative_target=2','--policy.temporal_ensemble_coeff=.025']);a.session_dir.mkdir()
    observed={};obs={k:50. for k in protocol.POSITION_FIELDS};obs.update({'gripper.load':1.,'gripper.curr':2.},top=np.zeros((480,640,3),np.uint8),gripper=np.zeros((480,640,3),np.uint8))
    class Robot:
        def __init__(self,cfg):observed['cfg']=cfg;self.is_connected=False
        def connect(self,calibrate=False):self.is_connected=True
        is_calibrated=True
        def get_observation(self):return obs
        def send_action(self,request):observed['sent']=request;return request
        def disconnect(self):observed['disconnect']=True;self.is_connected=False
    class Runtime:
        identity={};local=None;resets=0
        def reset(self):self.resets+=1
        def step(self,obs):
            if interrupt:raise KeyboardInterrupt
            return np.zeros(6),np.zeros(6),None
    class Writer:
        def isOpened(self):return True
        def write(self,x):pass
        def release(self):pass
    monkeypatch.setitem(sys.modules,'cv2',SimpleNamespace(cvtColor=lambda x,c:x,COLOR_RGB2BGR=1,imwrite=lambda *a:True,VideoWriter=lambda *a:Writer(),VideoWriter_fourcc=lambda *a:0))
    monkeypatch.setitem(sys.modules,'lerobot.robots.so_follower',SimpleNamespace(SO101Follower=Robot,SO101FollowerConfig=lambda **k:SimpleNamespace(**k)))
    monkeypatch.setitem(sys.modules,'lerobot.cameras.opencv.configuration_opencv',SimpleNamespace(OpenCVCameraConfig=lambda **k:k))
    monkeypatch.setitem(sys.modules,'lerobot.utils.control_utils',SimpleNamespace(init_keyboard_listener=lambda:(None,{'stop_recording':False,'exit_early':False})))
    commands=iter(['assist','end']);monkeypatch.setattr(control,'command',lambda:next(commands));monkeypatch.setattr(control.time,'sleep',lambda n:None)
    runtime=Runtime();control.run_trial(a,runtime,control.settings(a))
    path=a.session_dir/protocol.trial_id(a.model,a.position);summary=json.loads((path/'summary.json').read_text())
    assert summary['status']==('SAFETY_ABORT' if interrupt else 'OPERATOR_END')
    assert summary['full_autonomous_success']=='FAIL' and summary['assisted_recovery_used']
    assert observed['cfg'].max_relative_target==2 and observed['disconnect'] and runtime.resets==2
    assert not list(a.session_dir.rglob('*STOP*'))
    # Re-running the same trial must go through: the old attempt is moved to _prev_.
    commands=iter(['end']);monkeypatch.setattr(control,'command',lambda:next(commands))
    control.run_trial(a,runtime,control.settings(a))
    assert not list(a.session_dir.rglob('*STOP*'))
    archived=list(a.session_dir.glob(f'{path.name}_prev_*'))
    assert len(archived)==1 and (archived[0]/'summary.json').is_file()
    assert json.loads((path/'summary.json').read_text())['status']=='OPERATOR_END'
    assert json.loads((archived[0]/'summary.json').read_text())['status']==summary['status']
    with (a.session_dir/'trial_manifest.csv').open(newline='',encoding='utf-8') as f:
        rows=[r for r in csv.DictReader(f) if r['TRIAL_ID']==path.name]
    assert len(rows)==1 and rows[0]['status']=='OPERATOR_END'


def test_residual_metrics_and_payload_paths(tmp_path):
    r=control.residual_metrics([0.,1.,-1.,0.,1.]);assert r['nonzero_fraction']==.6 and r['sign_change_rate']==1
    assert control.residual_metrics([])['mean_abs'] is None
    with pytest.raises(ValueError):protocol.payload_path(tmp_path,'../outside')


def test_summary_preserves_incomplete_attempt_and_allows_downstream_recovery(tmp_path):
    spec=importlib.util.spec_from_file_location('summarize_f5',ROOT/'scripts/summarize_f5.py')
    summarize=importlib.util.module_from_spec(spec);spec.loader.exec_module(summarize)
    assert summarize.summarize(tmp_path)['status']=='NOT_RUN'
    a=args(tmp_path);identifier=protocol.trial_id(a.model,'A')
    record=dict(MODEL_ID=a.model,TRIAL_ID=identifier,POSITION='A',planned_slot=2,status='STARTED',rows=0)
    control.update_manifest(tmp_path,record)
    assert summarize.summarize(tmp_path)['status']=='INCOMPLETE_RUNTIME_EVIDENCE'
    folder=tmp_path/identifier;folder.mkdir()
    review=dict(MODEL_ID=a.model,TRIAL_ID=identifier,POSITION='A',A={**{f'S{i}':'UNKNOWN' for i in range(7)},'full_autonomous_success':'FAIL'},
        B={**{f'S{i}':('PASS' if i>=2 else 'UNKNOWN') for i in range(7)},'full_chain_after_assistance':'PASS'},
        validity='TASK_FAILURE',visible_slip='UNKNOWN',premature_release=False,visible_over_closure=False,cup_deformation=False,
        operator_emergency_stop=False,approximate_lift_height_cm=None,approximate_hold_duration_s=None,
        first_autonomous_failure_stage='S1',assistance_type='reposition cup')
    control.save(folder/'stage_review.json',review)
    control.save(folder/'summary.json',dict(**record,assisted_recovery_used=True))
    assert summarize.summarize(tmp_path)['status']=='REVIEW_REQUIRED'


def test_summary_ignores_superseded_prev_attempts(tmp_path):
    spec=importlib.util.spec_from_file_location('summarize_f5',ROOT/'scripts/summarize_f5.py')
    summarize=importlib.util.module_from_spec(spec);spec.loader.exec_module(summarize)
    a=args(tmp_path);identifier=protocol.trial_id(a.model,'A')
    stages={f'S{i}':'UNKNOWN' for i in range(7)}
    review=dict(MODEL_ID=a.model,TRIAL_ID=identifier,POSITION='A',A=dict(**stages,full_autonomous_success='UNKNOWN'),
        B=dict(**stages,full_chain_after_assistance='UNKNOWN'),validity='UNREVIEWED',visible_slip='UNKNOWN',
        premature_release='UNKNOWN',visible_over_closure='UNKNOWN',cup_deformation='UNKNOWN',operator_emergency_stop=False,
        approximate_lift_height_cm=None,approximate_hold_duration_s=None,first_autonomous_failure_stage='',assistance_type='')
    for name in [identifier,identifier+'_prev_20261022_120000']:
        folder=tmp_path/name;folder.mkdir()
        control.save(folder/'stage_review.json',review)
        control.save(folder/'summary.json',dict(MODEL_ID=a.model,TRIAL_ID=identifier,POSITION='A',assisted_recovery_used=False))
    assert summarize.summarize(tmp_path)['attempted_trials']==1
