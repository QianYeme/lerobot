"""One directly executable F5 trial. No Stage0/shadow/fingerprint/STOP locks."""
import argparse
import csv
import json
import math
import select
import sys
import time
import traceback
from pathlib import Path

from lerobot.utils.f5_protocol import MODELS, POSITION_FIELDS, SCHEDULE, BOUND, trial_id, slot


def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def parser(models=MODELS):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--session-dir',type=Path,required=True)
    p.add_argument('--model',choices=models,required=True);p.add_argument('--position',choices=['A','B','C'],required=True)
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--execute',action='store_true');mode.add_argument('--dry-run',action='store_true')
    for key,value,kind in [('robot.type','so101_follower',str),('robot.port','/dev/ttyACM0',str),('robot.id','nn',str),
        ('robot.max_relative_target',5.,float),('policy.device','cuda',str),('policy.n_action_steps',1,int),
        ('policy.temporal_ensemble_coeff',.01,float),('dataset.fps',30,int),('dataset.num_episodes',1,int),
        ('dataset.episode_time_s',60.,float),('dataset.reset_time_s',30.,float),('dataset.single_task','Cup pick and place',str)]:
        p.add_argument('--'+key,dest=key.replace('.','_'),type=kind,default=value)
    p.add_argument('--robot.calibration_dir',dest='calibration_dir',type=Path)
    p.add_argument('--robot.cameras',dest='camera_spec',default='{top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, gripper: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}}')
    return p


def settings(a):
    import yaml
    cameras=yaml.safe_load(a.camera_spec)
    if set(cameras)!=set(['top','gripper']):raise ValueError('Exactly top/gripper required')
    for c in cameras.values():
        if c.get('type')!='opencv' or (c.get('width'),c.get('height'),c.get('fps'))!=(640,480,30):raise ValueError('Use fixed RGB640x480/30 cameras')
        if set(c)-{'type','index_or_path','width','height','fps','fourcc'}:raise ValueError('Unexpected camera preprocessing')
        if c.get('fourcc','MJPG') not in ['MJPG','YUYV']:raise ValueError('Unsupported camera codec')
    if (a.robot_type,a.policy_n_action_steps,a.dataset_fps,a.dataset_num_episodes)!=(
        'so101_follower',1,30,1):raise ValueError('F5 requires SO101/action_steps1/fps30/one trial settings')
    if not math.isfinite(a.robot_max_relative_target) or a.robot_max_relative_target<=0:raise ValueError('max_relative_target must be finite and positive')
    if not math.isfinite(a.policy_temporal_ensemble_coeff):raise ValueError('temporal_ensemble_coeff must be finite')
    if (a.dataset_episode_time_s,a.dataset_reset_time_s)!=(60.,30.):raise ValueError('F5 fixed60s trial /30s manual reset allowance')
    return dict(cameras=cameras,robot_type=a.robot_type,robot_port=a.robot_port,robot_id=a.robot_id,
        max_relative_target=a.robot_max_relative_target,n_action_steps=a.policy_n_action_steps,
        temporal_ensemble_coeff=a.policy_temporal_ensemble_coeff,fps=a.dataset_fps,
        episode_time_s=a.dataset_episode_time_s,reset_time_s=a.dataset_reset_time_s,task=a.dataset_single_task)


def update_manifest(directory,record):
    path=directory/'trial_manifest.csv';rows=[]
    if path.exists():
        with path.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    rows=[r for r in rows if r['TRIAL_ID']!=record['TRIAL_ID']]+[record]
    rows.sort(key=lambda r:int(r['planned_slot']))
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(record));w.writeheader();w.writerows(rows)


def residual_metrics(values):
    import numpy as np
    if not values:return dict(samples=0,mean_abs=None,median_abs=None,max_abs=None,near_bound_fraction=None,sign_change_rate=None,nonzero_fraction=None)
    x=np.asarray(values,dtype=float);nonzero=x[x!=0]
    return dict(samples=len(x),mean_abs=float(np.abs(x).mean()),median_abs=float(np.median(np.abs(x))),max_abs=float(np.abs(x).max()),
        near_bound_fraction=float((np.abs(x)>=.95*BOUND).mean()),nonzero_fraction=float((x!=0).mean()),
        sign_change_rate=float((np.sign(nonzero[1:])!=np.sign(nonzero[:-1])).mean()) if len(nonzero)>1 else None,
        sign_change_definition='Fraction of adjacent nonzero corrections whose signs differ; exact0 omitted; not physical force')


def command():
    # Linux terminal input remains available when stdout is piped through tee.
    if sys.stdin.isatty() and select.select([sys.stdin],[],[],0)[0]:return sys.stdin.readline().strip().lower()
    return ''


def run_trial(a,runtime,common,protocol=None):
    import cv2
    import numpy as np
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SO101Follower,SO101FollowerConfig
    from lerobot.utils.control_utils import init_keyboard_listener
    get_id=protocol.trial_id if protocol else trial_id
    get_slot=protocol.slot if protocol else slot
    total_slots=len(protocol.SCHEDULE) if protocol else len(SCHEDULE)
    stage_prefix=protocol.STAGE_PREFIX if protocol else 'S'
    identifier=getattr(a,'trial_id_override',None) or get_id(a.model,a.position);directory=a.session_dir/identifier
    if Path(identifier).name!=identifier or identifier in ['.','..']:raise ValueError('trial-id must be a plain directory name')
    # Re-running a trial must not be blocked by the previous attempt. Move the old
    # directory aside (evidence kept, never deleted) and write the new attempt fresh.
    if directory.exists():
        stamp=time.strftime('%Y%m%d_%H%M%S');archived=a.session_dir/f'{identifier}_prev_{stamp}';suffix=1
        while archived.exists():
            archived=a.session_dir/f'{identifier}_prev_{stamp}_{suffix}';suffix+=1
        directory.rename(archived)
        print(f'Previous {identifier} moved to {archived.name}; starting a new attempt.',flush=True)
    directory.mkdir(exist_ok=False)
    record=dict(MODEL_ID=a.model,TRIAL_ID=identifier,POSITION=a.position,planned_slot=get_slot(a.model,a.position),
        status='STARTED',rows=0,assisted_recovery_used=False,control_csv=str(directory/f'{identifier}.csv'),
        top_video=str(directory/f'{identifier}_top.mp4'),gripper_video=str(directory/f'{identifier}_gripper.mp4'))
    if protocol is not None and hasattr(protocol,'audit_initial'):record['valid_initial_state']=None
    update_manifest(a.session_dir,record)
    save(directory/'configuration.json',dict(common=common,identity=runtime.identity,**record,started_wall_time=time.time(),
        state_order=runtime.identity.get('state_fields',POSITION_FIELDS+(['gripper.load','gripper.curr'] if a.model=='P1-Hist' else [])),
        local_application='Once after temporal ensemble+global denormalization, before robot relative target limit',
        stage_names=getattr(protocol,'STAGE_NAMES',None)))
    cfg=SO101FollowerConfig(port=a.robot_port,id=a.robot_id,max_relative_target=a.robot_max_relative_target,calibration_dir=a.calibration_dir,
        use_degrees=getattr(a,'robot_use_degrees',True),
        cameras={name:OpenCVCameraConfig(**{k:v for k,v in c.items() if k!='type'}) for name,c in common['cameras'].items()})
    robot=SO101Follower(cfg);listener=None;writers={};rows=0;phase='A';assistance_t=None;error=None;status='SETUP_ERROR';deltas=[];phase_deltas={'A':[],'B':[]};cleanup=[]
    print(f'{identifier} MODEL_ID={a.model} POSITION={a.position} planned_slot={get_slot(a.model,a.position)}/{total_slots}',flush=True)
    print('Home+Open、杯位与水量已固定；物理急停可用。Ctrl+C/Esc停止，右箭头正常结束。',flush=True)
    print('只有确认自主失败后，输入 assist 并回车，才能标记进入辅助B；输入 end 并回车结束。',flush=True)
    try:
        robot.connect(calibrate=False)
        if not robot.is_calibrated:raise ValueError('Existing matching calibration required; no automatic calibration')
        listener,events=init_keyboard_listener();runtime.reset()
        first=robot.get_observation()
        for camera in ['top','gripper']:
            if np.asarray(first[camera]).shape!=(480,640,3):raise ValueError('Camera shape mismatch')
            cv2.imwrite(str(directory/f'{identifier}_{camera}_first.jpg'),cv2.cvtColor(first[camera],cv2.COLOR_RGB2BGR))
            writer=cv2.VideoWriter(str(directory/f'{identifier}_{camera}.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(640,480))
            writers[camera]=writer
            if not writer.isOpened():raise ValueError('Video writer failed before first action')
        if protocol is not None and hasattr(protocol,'audit_initial'):
            initial=protocol.audit_initial(first,a,common);save(directory/'initial_state.json',initial)
            record['valid_initial_state']=initial['valid']
        fields=['MODEL_ID','TRIAL_ID','POSITION','phase','step','t_s','wall_time','inference_s','processing_s','overrun',
            'gripper.load','gripper.curr','nominal_gripper','delta_gripper','final_gripper']
        fields += [prefix+'.'+field for prefix in ['state','nominal','requested','sent'] for field in POSITION_FIELDS]
        with (directory/f'{identifier}.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();start=time.perf_counter();status='TIME_LIMIT'
            while time.perf_counter()-start<60.:
                if events['stop_recording']:status='SAFETY_ABORT';break
                if events['exit_early']:status='OPERATOR_END';break
                text=command()
                if text=='end':status='OPERATOR_END';break
                if text=='assist' and phase=='A':
                    phase='B';assistance_t=time.perf_counter()-start
                    save(directory/'assistance_event.json',dict(autonomous_failure_confirmed_by_operator=True,t_s=assistance_t,
                        MODEL_ID=a.model,TRIAL_ID=identifier,POSITION=a.position,policy_reset=False))
                    print('已确认A自主失败，进入B辅助；后续恢复不算自主成功。不要把手伸入运动区域。',flush=True)
                began=time.perf_counter();obs=robot.get_observation()
                for name,v in writers.items():v.write(cv2.cvtColor(obs[name],cv2.COLOR_RGB2BGR))
                inference_start=time.perf_counter();action,nominal,telemetry=runtime.step(obs);infer=time.perf_counter()-inference_start
                if not np.isfinite(action).all():raise ValueError('Nonfinite requested action')
                requested=dict(zip(POSITION_FIELDS,map(float,action)));sent=robot.send_action(requested)
                if not all(np.isfinite(sent[k]) for k in POSITION_FIELDS):raise ValueError('Nonfinite returned sent action')
                delta=float(telemetry['clipped_delta'][0]) if telemetry else 0.
                if runtime.local is not None:deltas.append(delta);phase_deltas[phase].append(delta)
                duration=time.perf_counter()-began
                row=dict(MODEL_ID=a.model,TRIAL_ID=identifier,POSITION=a.position,phase=phase,step=rows,t_s=began-start,wall_time=time.time(),
                    inference_s=infer,processing_s=duration,overrun=duration>1/30,nominal_gripper=float(nominal[5]),delta_gripper=delta,final_gripper=float(action[5]),
                    **{'gripper.load':obs.get('gripper.load'),'gripper.curr':obs.get('gripper.curr')})
                for prefix,values in [('state',obs),('nominal',dict(zip(POSITION_FIELDS,map(float,nominal)))),('requested',requested),('sent',sent)]:
                    row.update({prefix+'.'+k:values[k] for k in POSITION_FIELDS})
                writer.writerow(row);f.flush();rows+=1
                time.sleep(max(0,1/30-(time.perf_counter()-began)))
    except KeyboardInterrupt:status='SAFETY_ABORT';error='KeyboardInterrupt'
    except Exception:
        status='ERROR';error=traceback.format_exc();(directory/'error.log').write_text(error,encoding='utf-8');raise
    finally:
        for v in writers.values():
            try:v.release()
            except Exception:cleanup.append(traceback.format_exc())
        try:
            if listener is not None:listener.stop()
        except Exception:cleanup.append(traceback.format_exc())
        try:
            if robot.is_connected:robot.disconnect()
            else:
                for camera in robot.cameras.values():
                    if camera.is_connected:camera.disconnect()
                if robot.bus.is_connected:robot.bus.disconnect(disable_torque=True)
        except Exception:cleanup.append(traceback.format_exc())
        try:runtime.reset()
        except Exception:cleanup.append(traceback.format_exc())
        if cleanup:status='CLEANUP_ERROR'
        record.update(status=status,rows=rows,assisted_recovery_used=assistance_t is not None)
        save(directory/'summary.json',dict(**record,error=error,cleanup_errors=cleanup,assistance_t_s=assistance_t,
            full_autonomous_success='FAIL' if assistance_t is not None else 'UNKNOWN',
            termination_is_not_behavioral_success=True,video_clock='nominal30FPS, identity-level link only; no subsecond alignment claim'))
        if runtime.local is not None:save(directory/'local_regulator_metrics.json',dict(MODEL_ID=a.model,TRIAL_ID=identifier,POSITION=a.position,**residual_metrics(deltas),by_phase={k:residual_metrics(v) for k,v in phase_deltas.items()}))
        if not (directory/'stage_review.json').exists():
            template={f'{stage_prefix}{i}':'UNKNOWN' for i in range(7)}
            save(directory/'stage_review.json',dict(MODEL_ID=a.model,TRIAL_ID=identifier,POSITION=a.position,
                A=dict(**template,full_autonomous_success='FAIL' if assistance_t is not None else 'UNKNOWN'),
                B=dict(**template,phase_used=assistance_t is not None,full_chain_after_assistance='UNKNOWN'),
                visible_slip='UNKNOWN',premature_release='UNKNOWN',visible_over_closure='UNKNOWN',cup_deformation='UNKNOWN',
                operator_emergency_stop=status=='SAFETY_ABORT',assistance_type='',first_autonomous_failure_stage='',
                approximate_lift_height_cm=None,approximate_hold_duration_s=None,validity='UNREVIEWED',notes=''))
        update_manifest(a.session_dir,record)
        print(json.dumps(record),flush=True)
    if cleanup:raise RuntimeError('Cleanup error: inspect summary and physical emergency stop; no persistent STOP lock created')


def main():
    a=parser().parse_args();common=settings(a)
    import torch
    import numpy as np
    from lerobot.policies.act.f5_runtime import F5Runtime
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    if (a.bundle/'torch_cache').exists():torch.hub.set_dir(str(a.bundle/'torch_cache/hub'))
    a.session_dir.mkdir(parents=True,exist_ok=True)
    runtime=F5Runtime(a.bundle,a.model,a.policy_device,a.policy_n_action_steps,a.policy_temporal_ensemble_coeff)
    if a.dry_run:
        obs={k:0. for k in POSITION_FIELDS+['gripper.load','gripper.curr']};obs.update(top=np.zeros((480,640,3),dtype=np.uint8),gripper=np.zeros((480,640,3),dtype=np.uint8))
        action,_,_=runtime.step(obs);runtime.reset()
        save(a.session_dir/f'offline_{a.model}.json',dict(status='SYNTHETIC_CODE_SMOKE_ONLY',action=action.tolist(),common=common,hardware='NOT_CONNECTED'))
        return
    run_trial(a,runtime,common)


if __name__=='__main__':main()
