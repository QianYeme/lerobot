"""Start server-side HF upload; secret travels via SSH stdin, never a file."""
import os

import paramiko

client=paramiko.SSHClient();client.load_system_host_keys();client.set_missing_host_key_policy(paramiko.RejectPolicy())
client.connect(os.environ['D2_HOST'],port=int(os.environ['D2_PORT']),username='root',password=os.environ['D2_PASSWORD'],timeout=30)
try:
    command='''
test ! -e /root/autodl-tmp/R5_20261023/publish || exit 2
if test -f /etc/network_turbo; then . /etc/network_turbo >/dev/null 2>&1; fi
read -r HF_TOKEN
export HF_TOKEN
export HF_ENDPOINT=https://huggingface.co HF_HUB_DISABLE_XET=1
nohup /root/miniconda3/bin/python -u /root/autodl-tmp/R5_20261023/publish_r5_hub.py > /root/autodl-tmp/R5_20261023/publish.log 2>&1 < /dev/null &
echo UPLOAD_PID=$!
'''
    stdin,stdout,stderr=client.exec_command(command)
    stdin.write(os.environ['HF_TOKEN']+'\n');stdin.flush();stdin.channel.shutdown_write()
    print(stdout.read().decode());print(stderr.read().decode())
    if stdout.channel.recv_exit_status()!=0:raise RuntimeError('Upload launch failed; inspect existing publish state')
finally:client.close()
