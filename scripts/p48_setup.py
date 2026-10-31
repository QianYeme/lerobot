"""P4.8 G0 only: isolated dependencies and immutable official source; no training."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import argparse

ROOT=Path('/root/autodl-tmp/lerobot-main')
WORK=ROOT/'experiments/p4_public_benchmark/p4_8_libero_pilot/index1'
OUT=WORK/'result'
ENV=Path('/root/autodl-tmp/p48_isolated_env')


def write(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2)+'\n')


def run(command,**kwargs):
    return subprocess.check_output(command,text=True,stderr=subprocess.STDOUT,**kwargs)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'setup.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'setup_started.json').exists():
        if not args.resume:raise RuntimeError('Already registered; inspect existing setup before resuming')
        assert json.loads((OUT/'setup_complete.json').read_text())['status']=='ENV_BLOCKED'
        for name in ['dependency_install.log','setup_error.log','setup_complete.json','setup_started.json']:
            p=OUT/name
            if p.exists():shutil.copy2(p,OUT/(p.stem+'_attempt1'+p.suffix))
    write('setup_started.json',dict(pid=os.getpid(),started=time.time(),phase='G0',training=False))
    try:
        before=run([sys.executable,'-m','pip','list','--format=json'])
        (OUT/'base_packages_before.json').write_text(before)
        gpu=run(['nvidia-smi','--query-gpu=name,memory.total,memory.used,utilization.gpu','--format=csv,noheader'])
        jobs=run(['ps','-eo','pid,ppid,etime,comm'])
        write('server_resource_snapshot.json',dict(time=time.time(),gpu=gpu,processes=jobs,
             free_data_bytes=shutil.disk_usage(WORK).free,free_system_bytes=shutil.disk_usage('/').free,wall_clock_limit=None))
        assert shutil.disk_usage(WORK).free>20*1024**3
        parent=ROOT/'experiments/p4_final_p1_training/p4_5_robot_reliability_server10h/index1'
        source_manifest=json.loads((parent/'result/server/stock_source_manifest.json').read_text())
        for item in source_manifest['files']:
            path=parent/'stock'/item['path']
            assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
        shutil.copytree(parent/'stock',OUT/'workspace/lerobot',dirs_exist_ok=True)
        write('source_provenance.json',dict(lerobot_commit='8fff0fde7c79f23a93d845d1a50e985de01f8b8a',
             source=str(OUT/'workspace/lerobot'),source_verification='342 pinned official files checked',
             stock_manifest=source_manifest,so101_weights_reused=False,cup_prototypes_reused=False))
        if ENV.exists():
            assert args.resume and (ENV/'pyvenv.cfg').exists(),'Existing env is not an owned recovery target'
        else:subprocess.run([sys.executable,'-m','venv','--system-site-packages',str(ENV)],check=True)
        env=os.environ.copy();env['PIP_REQUIRE_VIRTUALENV']='true';env['PYTHONPATH']=str(OUT/'workspace/lerobot/src')
        for name in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']:env.pop(name,None)
        py=str(ENV/'bin/python')
        command=[py,'-m','pip','install','--index-url','https://pypi.tuna.tsinghua.edu.cn/simple','--report',str(OUT/'dependency_install_report.json'),'hf-libero==0.1.3']
        with (OUT/'dependency_install.log').open('w') as log:
            subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        after=run([sys.executable,'-m','pip','list','--format=json'])
        (OUT/'base_packages_after.json').write_text(after)
        assert json.loads(before)==json.loads(after),'Base package metadata changed'
        (OUT/'isolated_pip_freeze.txt').write_text(run([py,'-m','pip','freeze'],env=env))
        code="import sys,inspect,torch,lerobot,importlib.metadata as m; print('PREFIX',sys.prefix); print('LEROBOT',lerobot.__file__); print('TORCH',torch.__version__); print('HF_LIBERO',m.version('hf-libero')); print('CUDA',torch.cuda.is_available())"
        probe=run([py,'-c',code],env=env);(OUT/'isolation_probe.txt').write_text(probe);print(probe,flush=True)
        assert str(ENV) in probe and str(OUT/'workspace/lerobot/src') in probe
        (OUT/'environment_lock.md').write_text('# Isolated P4.8 G0\n\n'+probe+'\nRead-only inherited base Torch; simulator dependencies installed inside the venv only. Base pip metadata unchanged. Full versions and wheel URLs/hashes: isolated_pip_freeze.txt and dependency_install_report.json. No P4.7 environment upgrade. No training.\n')
        write('setup_complete.json',dict(status='G0_ENVIRONMENT_READY_SOURCE_DATA_PENDING',finished=time.time(),environment=str(ENV),training=False))
    except Exception:
        import traceback
        (OUT/'setup_error.log').write_text(traceback.format_exc())
        write('setup_complete.json',dict(status='ENV_BLOCKED',finished=time.time(),training=False))
        raise


if __name__=='__main__':main()
