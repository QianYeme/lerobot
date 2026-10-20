"""One F3-r1 trial. Default is NO hardware access; --execute needs human reviews."""
import argparse
import csv
import hashlib
import json
import platform
import time
import traceback
from pathlib import Path

from lerobot.utils.f3_control_protocol import check_review, check_trial, fingerprint


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--model', choices=['M0','M1','M2','M3'], default='M1')
    p.add_argument('--mode', choices=['disabled','shadow','regulated'], default='disabled')
    p.add_argument('--session-dir', type=Path, required=True)
    p.add_argument('--trial-id')
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare-only', action='store_true')
    modes.add_argument('--dry-run', action='store_true')
    modes.add_argument('--execute', action='store_true')
    p.add_argument('--stage0-report', type=Path)
    p.add_argument('--shadow-review', type=Path)
    for key, default, kind in [('robot.type','so101_follower',str),('robot.port','/dev/ttyACM0',str),
        ('robot.id','nn',str),('robot.max_relative_target',5.,float),('policy.device','cuda',str),
        ('policy.n_action_steps',1,int),('dataset.fps',30,int),('dataset.num_episodes',1,int),
        ('dataset.episode_time_s',60.,float),('dataset.reset_time_s',30.,float),
        ('dataset.single_task','Cup pick and place',str)]:
        p.add_argument('--'+key,dest=key.replace('.','_'),default=default,type=kind)
    p.add_argument('--policy.temporal_ensemble_coeff',dest='ensemble',type=float,default=None)
    p.add_argument('--robot.calibration_dir',dest='calibration_dir',type=Path)
    p.add_argument('--robot.cameras',dest='camera_spec',default='{top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, gripper: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}')
    return p


def settings(args):
    import yaml
    from lerobot.utils.constants import HF_LEROBOT_CALIBRATION
    cameras = yaml.safe_load(args.camera_spec)
    if set(cameras) != {'top','gripper'}:
        raise ValueError('Exactly top/gripper cameras required')
    for camera in cameras.values():
        if camera.get('type') != 'opencv' or (camera.get('width'),camera.get('height'),camera.get('fps')) != (640,480,30):
            raise ValueError('Frozen RGB640x480/30 OpenCV camera semantics required')
        if set(camera)-{'type','index_or_path','width','height','fps'}:
            raise ValueError('Do not silently alter camera color/rotation/processing in F3')
    if args.robot_type != 'so101_follower' or args.policy_n_action_steps != 1 or args.dataset_num_episodes != 1:
        raise ValueError('F3 requires SO101/n_action_steps1/one trial per invocation')
    if args.robot_max_relative_target <= 0 or args.dataset_fps <= 0 or args.dataset_episode_time_s <= 0:
        raise ValueError('Positive safety limit/frequency/time required')
    calibration = (args.calibration_dir or HF_LEROBOT_CALIBRATION/'robots'/'so_follower')/f'{args.robot_id}.json'
    source = Path(__file__).resolve().parents[1]
    code = [Path(__file__),source/'policies/act/f3_runtime.py',source/'policies/act/local_load_regulator.py',
            source/'utils/f3_control_protocol.py',source/'policies/act/modeling_act.py',
            source/'policies/act/configuration_act.py',source/'policies/act/object_conditioning/modeling_object_conditioning.py',
            source/'robots/so_follower/so_follower.py',source/'robots/so_follower/config_so_follower.py']
    manifest = json.loads((args.bundle/'manifest.json').read_text())['sha256']
    critical = ['model_identity.json','normalization.json','local_normalization.json','assets/cup_prototype_bank_v1.pt',
        'global/config.json','models/M0/config.json','models/M2/config.json','local/010000.pt',
        'models/M0/policy_preprocessor.json','models/M0/policy_postprocessor.json',
        'models/M0/policy_preprocessor_step_3_normalizer_processor.safetensors',
        'models/M0/policy_postprocessor_step_0_unnormalizer_processor.safetensors']
    payload = {name:hashlib.sha256((args.bundle/name).read_bytes()).hexdigest() for name in critical}
    if any(manifest.get(name) != value for name,value in payload.items()):
        raise ValueError('Frozen normalization/config/local/prototype identity mismatch')
    return dict(bundle=str(args.bundle.resolve()),model_identity_sha256=hashlib.sha256((args.bundle/'model_identity.json').read_bytes()).hexdigest(),
        host=platform.node(),robot_type=args.robot_type,port=args.robot_port,robot_id=args.robot_id,
        calibration_path=str(calibration.resolve()),calibration_sha256=hashlib.sha256(calibration.read_bytes()).hexdigest() if calibration.is_file() else None,
        cameras=cameras,max_relative_target=args.robot_max_relative_target,n_action_steps=1,ensemble=args.ensemble,
        fps=args.dataset_fps,episode_time_s=args.dataset_episode_time_s,reset_time_s=args.dataset_reset_time_s,
        task=args.dataset_single_task,device=args.policy_device,tf32=True,use_amp=False,
        payload_sha256=payload,
        code_sha256={str(path.relative_to(source)):hashlib.sha256(path.read_bytes()).hexdigest() for path in code})


def check_execution(args, common, key):
    check_trial(args.model,args.mode,args.trial_id or '')
    if not common['calibration_sha256']:
        raise ValueError('Existing matching calibration file required; no automatic calibration')
    if not args.stage0_report:
        raise ValueError('--stage0-report required for hardware execution')
    check_review(json.loads(args.stage0_report.read_text()),key)
    if args.model == 'M3' and args.mode == 'regulated':
        if (args.session_dir/'M3_STOP.json').exists():
            raise ValueError('M3 intervention stopped for this session')
        if not args.shadow_review:
            raise ValueError('Three shadow trials must be reviewed before M3 intervention')
        review = json.loads(args.shadow_review.read_text())
        check_review(review,key,shadow=True)
        for trial in review['valid_trial_ids']:
            directory = args.session_dir/trial
            summary = json.loads((directory/'summary.json').read_text())
            if summary.get('deployment_fingerprint') != key or summary.get('model') != 'M3' or summary.get('mode') != 'shadow' or summary.get('rows',0) <= 0 or summary.get('status') not in ['TIME_LIMIT','OPERATOR_END']:
                raise ValueError('Shadow evidence missing/aborted')
            for name in ['control.csv','top.mp4','gripper.mp4']:
                if not (directory/name).is_file() or (directory/name).stat().st_size == 0:
                    raise ValueError('Incomplete shadow logs/video')


def run_trial(args, runtime, common, key):
    import cv2
    import numpy as np
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from lerobot.policies.act.f3_runtime import POSITION_FIELDS
    from lerobot.utils.control_utils import init_keyboard_listener
    directory = args.session_dir/args.trial_id
    directory.mkdir(exist_ok=False)
    save(directory/'configuration.json',dict(common=common,fingerprint=key,model=args.model,mode=args.mode,
        identity=runtime.identity,started_wall_time=time.time()))
    cfg = SO101FollowerConfig(port=args.robot_port,id=args.robot_id,max_relative_target=args.robot_max_relative_target,
        calibration_dir=args.calibration_dir,cameras={name:OpenCVCameraConfig(**{k:v for k,v in camera.items() if k!='type'}) for name,camera in common['cameras'].items()})
    robot = SO101Follower(cfg)
    listener = None; writers = {}; rows = 0; status = 'SETUP_ERROR'; error = None
    try:
        if input(f'Physical stop ready, Home+Open and area clear? Type {args.trial_id} to connect: ').strip() != args.trial_id:
            raise ValueError('Operator did not confirm')
        robot.connect(calibrate=False)
        if not robot.is_calibrated:
            raise ValueError('Robot calibration mismatch; no auto-calibration')
        listener,events = init_keyboard_listener()
        runtime.reset()
        first = robot.get_observation()
        for name in ['top','gripper']:
            if np.asarray(first[name]).shape != (480,640,3):
                raise ValueError('Camera view shape mismatch')
            cv2.imwrite(str(directory/f'{name}_first.jpg'),cv2.cvtColor(first[name],cv2.COLOR_RGB2BGR))
            writer = cv2.VideoWriter(str(directory/f'{name}.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),args.dataset_fps,(640,480))
            writers[name] = writer
            if not writer.isOpened():
                raise ValueError('Video writer failed; no action sent')
        if input('Inspect both saved first-view images. Type START to send actions: ').strip() != 'START':
            raise ValueError('Views not approved')
        fields = ['step','t_s','wall_time','inference_s','processing_s','overrun','gripper.load','gripper.curr',
            'predicted_delta','clipped_delta','hypothetical_gripper']
        for prefix in ['state','nominal','requested','sent']:
            fields += [prefix+'.'+name for name in POSITION_FIELDS]
        with (directory/'control.csv').open('w',newline='') as stream:
            csv_writer = csv.DictWriter(stream,fieldnames=fields);csv_writer.writeheader()
            start = time.perf_counter();status = 'TIME_LIMIT'
            while time.perf_counter()-start < args.dataset_episode_time_s:
                if events['stop_recording']:
                    status = 'SAFETY_ABORT';break
                if events['exit_early']:
                    status = 'OPERATOR_END';break
                began = time.perf_counter();obs = robot.get_observation()
                if not np.isfinite(float(obs.get('gripper.load',float('nan')))):
                    raise ValueError('Missing/nonfinite raw load telemetry')
                for name,writer in writers.items():
                    writer.write(cv2.cvtColor(obs[name],cv2.COLOR_RGB2BGR))
                inference_start = time.perf_counter()
                action,nominal,telemetry = runtime.step(obs,args.mode)
                inference_s = time.perf_counter()-inference_start
                requested = dict(zip(POSITION_FIELDS,map(float,action)))
                sent = robot.send_action(requested)
                if not all(np.isfinite(sent[name]) for name in POSITION_FIELDS):
                    raise ValueError('Nonfinite sent action')
                duration = time.perf_counter()-began
                row = dict(step=rows,t_s=began-start,wall_time=time.time(),inference_s=inference_s,
                    processing_s=duration,overrun=duration>1/args.dataset_fps,
                    **{'gripper.load':obs.get('gripper.load'),'gripper.curr':obs.get('gripper.curr')},
                    predicted_delta=float(telemetry['predicted_delta'][0]) if telemetry else None,
                    clipped_delta=float(telemetry['clipped_delta'][0]) if telemetry else None,
                    hypothetical_gripper=float(telemetry['hypothetical_gripper'][0]) if telemetry else None)
                for prefix,values in [('state',obs),('nominal',dict(zip(POSITION_FIELDS,map(float,nominal)))),('requested',requested),('sent',sent)]:
                    row.update({prefix+'.'+name:values[name] for name in POSITION_FIELDS})
                csv_writer.writerow(row);stream.flush();rows += 1
                time.sleep(max(0,1/args.dataset_fps-(time.perf_counter()-began)))
    except KeyboardInterrupt:
        status = 'SAFETY_ABORT';error = 'KeyboardInterrupt'
    except Exception:
        status = 'ERROR';error = traceback.format_exc()
        (directory/'error.log').write_text(error)
        raise
    finally:
        summary = dict(status=status,error=error,rows=rows,model=args.model,mode=args.mode,
            deployment_fingerprint=key,video_clock='nominal-FPS playback; use CSV step and t_s for real timing')
        if args.model=='M3' and args.mode=='regulated' and status in ['ERROR','SAFETY_ABORT']:
            save(args.session_dir/'M3_STOP.json',summary)
        cleanup_error = None
        try:
            for writer in writers.values():writer.release()
            if listener is not None:listener.stop()
        except Exception:
            cleanup_error = traceback.format_exc()
        try:
            if robot.is_connected:robot.disconnect()
            else:
                try:
                    for cam in robot.cameras.values():
                        if cam.is_connected:cam.disconnect()
                finally:
                    if robot.bus.is_connected:robot.bus.disconnect(disable_torque=True)
        except Exception:
            cleanup_error = (cleanup_error or '')+traceback.format_exc()
        finally:
            if cleanup_error:
                summary.update(status='CLEANUP_ERROR',cleanup_error=cleanup_error)
                if args.model=='M3' and args.mode=='regulated':
                    save(args.session_dir/'M3_STOP.json',summary)
            runtime.reset();save(directory/'summary.json',summary)
        if cleanup_error:
            raise RuntimeError('Cleanup failed; inspect summary.json and physical robot safety')


def main():
    args = parser().parse_args()
    common = settings(args);key = fingerprint(common)
    args.session_dir.mkdir(parents=True,exist_ok=True)
    frozen = args.session_dir/'deployment_config_frozen.json'
    if frozen.exists():
        if json.loads(frozen.read_text()) != common:
            raise ValueError('Session deployment configuration changed; do not retune inside F3-r1')
    elif args.prepare_only:
        save(frozen,common)
    if args.prepare_only:
        for name in ['stage0_review.json','shadow_review.json']:
            path = args.session_dir/name
            if not path.exists():save(path,dict(passed=False,reviewer='',deployment_fingerprint=key,valid_trial_ids=[]))
        print('PREPARED_NO_HARDWARE',key);return
    if args.execute:
        if not frozen.exists():raise ValueError('Run --prepare-only and complete human Stage0 review first')
        check_execution(args,common,key)
    import torch
    import numpy as np
    from lerobot.policies.act.f3_runtime import F3Runtime, POSITION_FIELDS
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.benchmark=True
    runtime = F3Runtime(args.bundle,args.model,args.policy_device,args.policy_n_action_steps,args.ensemble)
    if args.dry_run:
        obs = {k:0. for k in POSITION_FIELDS+['gripper.load','gripper.curr']}
        obs.update(top=np.zeros((480,640,3),dtype=np.uint8),gripper=np.zeros((480,640,3),dtype=np.uint8))
        action,_,_ = runtime.step(obs,args.mode)
        save(args.session_dir/f'dry_run_{args.model}_{args.mode}.json',dict(status='OFFLINE_PASS',action=action.tolist(),hardware='NOT_CONNECTED'))
        print('OFFLINE_PASS_NO_HARDWARE',args.model,args.mode);return
    run_trial(args,runtime,common,key)


if __name__=='__main__':main()
