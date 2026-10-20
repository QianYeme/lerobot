"""Run F3-r1 artifact publication detached; credentials through SSH stdin only."""
import os
from datetime import datetime
import paramiko

client=paramiko.SSHClient();client.load_system_host_keys();client.set_missing_host_key_policy(paramiko.RejectPolicy())
client.connect(os.environ['D2_HOST'],port=int(os.environ['D2_PORT']),username='root',password=os.environ['D2_PASSWORD'],timeout=30)
try:
    logfile='publish_'+datetime.now().strftime('%Y%m%dT%H%M%S')+'.log'
    command='''. /etc/network_turbo >/dev/null 2>&1
read -r HF_TOKEN
export HF_TOKEN
export HF_ENDPOINT=https://huggingface.co
export HF_HUB_DISABLE_XET=1
nohup bash /root/autodl-tmp/F3_r1_transfer_20261020/publish_f3_r1.sh > /root/autodl-tmp/F3_r1_transfer_20261020/LOG_NAME 2>&1 < /dev/null &
echo PUBLISH_PID=$!
echo LOG_NAME
'''.replace('LOG_NAME',logfile)
    stdin,stdout,stderr=client.exec_command(command)
    stdin.write(os.environ['HF_TOKEN']+'\n');stdin.flush();stdin.channel.shutdown_write()
    print(stdout.read().decode());print(stderr.read().decode())
finally:client.close()
