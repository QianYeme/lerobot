"""Run a supplied command on the explicitly configured D2 server."""
import argparse
import os
import sys
from pathlib import Path
import paramiko

ap = argparse.ArgumentParser()
ap.add_argument('--command', required=True)
ap.add_argument('--upload', nargs=2, metavar=('LOCAL', 'REMOTE'))
ap.add_argument('--download', nargs=2, metavar=('REMOTE', 'LOCAL'))
ap.add_argument('--download-npz', nargs=2, metavar=('REMOTE_DIR', 'LOCAL_DIR'))
args = ap.parse_args()
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.RejectPolicy())
client.connect(os.environ['D2_HOST'], port=int(os.environ['D2_PORT']),
               username='root', password=os.environ['D2_PASSWORD'], timeout=30,
               banner_timeout=30, auth_timeout=30)
try:
    if args.upload:
        with client.open_sftp() as sftp:
            sftp.put(*args.upload)
    if args.download:
        with client.open_sftp() as sftp:
            sftp.get(*args.download)
    if args.download_npz:
        remote_dir,local_dir=args.download_npz
        Path(local_dir).mkdir(parents=True,exist_ok=True)
        with client.open_sftp() as sftp:
            for filename in sorted(sftp.listdir(remote_dir)):
                if filename.endswith('.npz'):
                    sftp.get(remote_dir+'/'+filename,str(Path(local_dir)/filename))
    _, stdout, stderr = client.exec_command(args.command, timeout=45)
    print(stdout.read().decode('utf-8', errors='replace'), end='')
    print(stderr.read().decode('utf-8', errors='replace'), end='', file=sys.stderr)
    status = stdout.channel.recv_exit_status()
finally:
    client.close()
sys.exit(status)
