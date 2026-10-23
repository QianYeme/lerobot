"""R5 protocol, direct CLI and existing mock-I/O loop regression; no hardware."""
import importlib.util
import ast
import json
import re
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from lerobot.utils import r5_protocol as protocol
from lerobot.scripts import control_r5
from lerobot.scripts import control_f5


def test_fixed_fifteen_slots_and_seed4():
    assert len(protocol.SCHEDULE)==15 and len(set(protocol.SCHEDULE))==15
    assert [m for p,m in protocol.SCHEDULE if p=='A']==['M1','M2','M3','M4','M5']
    assert [m for p,m in protocol.SCHEDULE if p=='B']==['M2','M3','M4','M5','M1']
    assert [m for p,m in protocol.SCHEDULE if p=='C']==['M3','M4','M5','M1','M2']
    assert protocol.LABELS['M5']=='G60-BadSeed-S4'
    assert protocol.EXPECTED['M1'][0]==protocol.EXPECTED['M3'][0]
    assert protocol.EXPECTED['M2'][0]==protocol.EXPECTED['M4'][0]
    assert protocol.trial_id('M4','C')=='R5_M4_C_01'


def test_r5_uses_r4_flags_and_cli_tunables(tmp_path,monkeypatch):
    calls=[]
    class Runtime:
        def __init__(self,*args):calls.extend(args)
        def step(self,obs):return np.zeros(6),np.zeros(6),None
        def reset(self):pass
    backend=SimpleNamespace(cudnn=SimpleNamespace(),cuda=SimpleNamespace(matmul=SimpleNamespace()))
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(set_num_threads=lambda n:None,backends=backend))
    monkeypatch.setitem(sys.modules,'lerobot.policies.act.r5_runtime',SimpleNamespace(R5Runtime=Runtime))
    monkeypatch.setattr(sys,'argv',['control_r5','--bundle',str(tmp_path/'bundle'),'--session-dir',str(tmp_path/'session'),
        '--model','M4','--position','A','--dry-run','--policy.temporal_ensemble_coeff=.025','--robot.max_relative_target=2'])
    control_r5.main()
    assert calls[-2:]==[1,.025] and backend.cudnn.benchmark is False
    assert backend.cuda.matmul.allow_tf32 and backend.cudnn.allow_tf32
    output=json.loads((tmp_path/'session/offline_M4.json').read_text())
    assert output['common']['max_relative_target']==2


def test_five_guide_commands_are_real_cli_and_editable():
    path=ROOT/'experiments/force/r5_pre_final_dataset_real_robot_gate/index1/REAL_ROBOT_TEST_GUIDE.md'
    text=path.read_text(encoding='utf-8')
    commands=re.findall(r'python src/lerobot/scripts/control_r5\.py (.*?)(?=\n```)',text,re.S)
    commands=[c for c in commands if '--execute' in c]
    assert len(commands)==5
    models=[]
    for command in commands:
        command=command.replace('\\\n',' ').split('2>&1')[0]
        command=re.sub(r'\$\{R5_POSITION[^}]*\}','A',command)
        command=command.replace('--policy.temporal_ensemble_coeff=0.01','--policy.temporal_ensemble_coeff=0.025')
        command=command.replace('--robot.max_relative_target=5','--robot.max_relative_target=2')
        args=control_f5.parser(protocol.MODELS).parse_args(shlex.split(command))
        config=control_f5.settings(args)
        assert config['temporal_ensemble_coeff']==.025 and config['max_relative_target']==2
        models.append(args.model)
    assert models==protocol.MODELS


def test_mock_trial_writes_r5_id_and_P_stages(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('f5_mock_tests',ROOT/'tests/scripts/test_f5_control.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    # The existing loop's fake-I/O regression exercises reset, assistance, rerun archive and Ctrl+C.
    original=module.control.run_trial
    def route(a,runtime,common):
        a.model='M4'
        return original(a,runtime,common,protocol=protocol)
    monkeypatch.setattr(module.control,'run_trial',route)
    monkeypatch.setattr(module.protocol,'trial_id',protocol.trial_id)
    module.test_mock_trial_assistance_or_ctrl_c_cleanup_no_stop_lock(tmp_path,monkeypatch,False)
    review=json.loads((tmp_path/'session'/protocol.trial_id('M4','A')/'stage_review.json').read_text())
    assert all(f'P{i}' in review['A'] for i in range(7))


def test_actual_r5_prepare_uses_float32_load_arithmetic():
    source=ast.parse((ROOT/'src/lerobot/policies/act/r5_runtime.py').read_text(encoding='utf-8'))
    original=next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name=='R5Runtime')
    method=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name=='prepare')
    tree=ast.Module(body=[ast.ClassDef(name='Probe',bases=[ast.Name(id='Base',ctx=ast.Load())],keywords=[],body=[method],decorator_list=[])],type_ignores=[])
    class Base:
        def prepare(self,observation):return {}
    fake_torch=SimpleNamespace(tensor=lambda value,**kw:np.asarray(value,np.float32),float32=np.float32)
    scope=dict(Base=Base,np=np,torch=fake_torch)
    exec(compile(ast.fix_missing_locations(tree),'<actual R5 prepare>','exec'),scope)
    model=scope['Probe']();model.device='cpu';model.norm={'load':{'mean':-4.73182782977835,'std':119.8893304407713}}
    for value in [-10.,0.,100.]:
        actual=model.prepare({'gripper.load':value})['observation.load'][0,0]
        expected=(np.float32(value)-np.float32(model.norm['load']['mean']))/np.float32(model.norm['load']['std'])
        assert actual==expected


def test_summary_keeps_unseen_and_unreviewed_explicit(tmp_path):
    spec=importlib.util.spec_from_file_location('summarize_r5',ROOT/'scripts/summarize_r5.py')
    summary=importlib.util.module_from_spec(spec);spec.loader.exec_module(summary)
    result=summary.summarize(tmp_path)
    assert result['status']=='NOT_RUN' and result['R5_COMPLETE'] is False
    import csv
    with (tmp_path/'positioning_summary.csv').open() as stream:rows=list(csv.DictReader(stream))
    assert len(rows)==35 and all(r['unrecorded_planned']=='3' and r['valid_trials']=='0' for r in rows)
