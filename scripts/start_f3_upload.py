"""Start official-HF upload via AutoDL turbo; secret sent on SSH stdin, never saved."""
import os
from datetime import datetime

import paramiko

client=paramiko.SSHClient();client.load_system_host_keys();client.set_missing_host_key_policy(paramiko.RejectPolicy())
client.connect(os.environ['D2_HOST'],port=int(os.environ['D2_PORT']),username='root',password=os.environ['D2_PASSWORD'],timeout=30)
try:
    logfile='upload_'+datetime.now().strftime('%Y%m%dT%H%M%S')+'.log'
    command="""pgrep -f '^/root/miniconda3/bin/python /root/autodl-tmp/F3_transfer_20261020/publish_f3_bundle.py --mode upload$' | xargs -r kill
. /etc/network_turbo >/dev/null 2>&1
read -r HF_TOKEN
export HF_TOKEN
export HF_ENDPOINT=https://huggingface.co
export HF_HUB_DISABLE_XET=1
nohup /root/miniconda3/bin/python -u /root/autodl-tmp/F3_transfer_20261020/publish_f3_bundle.py --mode upload > /root/autodl-tmp/F3_transfer_20261020/LOG_NAME 2>&1 < /dev/null &
echo UPLOAD_PID=$!
echo LOG_NAME
""".replace('LOG_NAME',logfile)
    stdin,stdout,stderr=client.exec_command(command)
    stdin.write(os.environ['HF_TOKEN']+'\n');stdin.flush();stdin.channel.shutdown_write()
    print(stdout.read().decode());print(stderr.read().decode())
finally:client.close()
