"""Exercise video/logging/shadow/failure cleanup using fake robot only, no serial."""
import builtins
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np

BASE=Path('/root/autodl-tmp/F3_control_validation')
spec=importlib.util.spec_from_file_location('control_f3',BASE/'source/src/lerobot/scripts/control_f3.py')
control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control)
from lerobot.policies.act.f3_runtime import POSITION_FIELDS


class FakeRobot:
    last=None
    def __init__(self,cfg):
        FakeRobot.last=self;self.is_connected=False;self.is_calibrated=True;self.sent=[]
        self.cameras={};self.bus=SimpleNamespace(is_connected=False)
    def connect(self,calibrate=False):
        assert calibrate is False;self.is_connected=True
    def get_observation(self):
        obs={k:0. for k in POSITION_FIELDS+['gripper.load','gripper.curr']}
        obs.update(top=np.zeros((480,640,3),dtype=np.uint8),gripper=np.zeros((480,640,3),dtype=np.uint8))
        return obs
    def send_action(self,action):self.sent.append(action.copy());return action.copy()
    def disconnect(self):self.is_connected=False


class FakeRuntime:
    identity={'mock':True}
    def __init__(self,fail=False):self.fail=fail;self.resets=0
    def reset(self):self.resets+=1
    def step(self,obs,mode):
        if self.fail:raise ValueError('injected inference failure')
        action=np.arange(6,dtype=np.float32)
        return action,action.copy(),None


def main():
    fake=ModuleType('lerobot.robots.so_follower');fake.SO101Follower=FakeRobot
    fake.SO101FollowerConfig=lambda **kw:SimpleNamespace(**kw)
    sys.modules['lerobot.robots.so_follower']=fake
    import lerobot.utils.control_utils
    lerobot.utils.control_utils.init_keyboard_listener=lambda:(None,dict(stop_recording=False,exit_early=False))
    original_input=builtins.input
    results={}
    try:
        with tempfile.TemporaryDirectory(prefix='f3_robot_mock_',dir=BASE) as folder:
            session=Path(folder)
            for fail in [False,True]:
                trial='R01_M0' if not fail else 'R01_M3'
                replies=iter([trial,'START']);builtins.input=lambda prompt:next(replies)
                args=SimpleNamespace(session_dir=session,trial_id=trial,model='M0' if not fail else 'M3',
                    mode='disabled' if not fail else 'regulated',robot_port='FAKE_ONLY',robot_id='nn',
                    robot_max_relative_target=5,calibration_dir=None,dataset_fps=30,dataset_episode_time_s=.03)
                common=dict(cameras={name:dict(type='opencv',index_or_path=idx,width=640,height=480,fps=30)
                    for name,idx in [('top',4),('gripper',2)]})
                runtime=FakeRuntime(fail)
                try:control.run_trial(args,runtime,common,'mock_fingerprint')
                except ValueError:
                    assert fail
                summary=json.loads((session/trial/'summary.json').read_text())
                assert not FakeRobot.last.is_connected and runtime.resets==2
                if fail:
                    assert summary['status']=='ERROR' and len(FakeRobot.last.sent)==0
                    assert (session/'M3_STOP.json').exists()
                else:
                    assert summary['rows']>=1 and len(FakeRobot.last.sent)==summary['rows']
                    for name in ['control.csv','top.mp4','gripper.mp4']:
                        assert (session/trial/name).stat().st_size>0
                results[trial]=summary
    finally:builtins.input=original_input
    (BASE/'mock_smoke.json').write_text(json.dumps(dict(status='PASS',fake_robot_only=True,trials=results),indent=2))
    print('PASS: fake robot/video/CSV/cleanup; M3 failure stop latch; no real hardware imported or opened')


if __name__=='__main__':main()
