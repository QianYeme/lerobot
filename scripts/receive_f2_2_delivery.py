"""Optional local receiver: credentials only inherited from environment, never saved."""
import hashlib
import json
import os
import shutil
import time
import zipfile
from pathlib import Path

import paramiko

ROOT=Path(__file__).resolve().parents[1]
INDEX=ROOT/'experiments/force/f2_2_contact_gated_single_frame_load/index1'
REMOTE='/root/autodl-tmp/F2_2_r0_20261019'


def main():
    deadline=time.monotonic()+24*3600
    while time.monotonic()<deadline:
        client=paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.RejectPolicy())
        try:
            client.connect(os.environ['D2_HOST'],port=int(os.environ['D2_PORT']),username='root',
                password=os.environ['D2_PASSWORD'],timeout=30,banner_timeout=30,auth_timeout=30)
            with client.open_sftp() as sftp:
                sftp.get(REMOTE+'/autofinish_state.json',str(INDEX/'result/AUTO_DELIVERY_STATE.json'))
                state=json.loads((INDEX/'result/AUTO_DELIVERY_STATE.json').read_text())
                if state['stage']=='BLOCKED':
                    print('Server stopped with preserved failure evidence; see AUTO_DELIVERY_STATE.json',flush=True)
                    return
                if state['stage']=='READY_STOPPED':
                    ready=state['ready']
                    package=INDEX/'F2_2_result.zip'
                    sftp.get(REMOTE+'/F2_2_result.zip',str(package))
                    assert hashlib.sha256(package.read_bytes()).hexdigest()==ready['package_sha256']
                    with zipfile.ZipFile(package) as archive:
                        assert archive.testzip() is None
                        for member in archive.infolist():
                            target=(INDEX/member.filename).resolve()
                            assert target.is_relative_to(INDEX.resolve()), 'Unsafe ZIP path'
                            assert member.filename.startswith('result/'), 'Unexpected ZIP root'
                        backup=INDEX/'pretraining_delivery_archive'
                        if not backup.exists():
                            shutil.copytree(INDEX/'result',backup)
                        archive.extractall(INDEX)
                    print('READY: verified ZIP received and result extracted; original pretraining reports archived.',flush=True)
                    return
        except (OSError,EOFError,paramiko.SSHException) as error:
            # Network interruption affects delivery only, never server training.
            print(type(error).__name__+': delivery connection unavailable; retry in45s',flush=True)
        finally:
            client.close()
        time.sleep(45)
    print('Receiver24h window ended; server jobs unaffected.',flush=True)


if __name__=='__main__':
    main()
