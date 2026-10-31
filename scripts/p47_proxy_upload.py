"""Temporary SSH proxy for an authenticated release; credentials only from environment."""
import os
import select
import socket
import threading
import time

import paramiko


def main():
    client=paramiko.SSHClient();client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(os.environ['P47_SSH_HOST'],port=int(os.environ['P47_SSH_PORT']),username='root',password=os.environ['P47_SSH_PASSWORD'],timeout=20)
    transport=client.get_transport();port=transport.request_port_forward('127.0.0.1',0)
    def bridge(channel):
        sock=None
        try:
            sock=socket.create_connection(('127.0.0.1',7890),timeout=120)
            while True:
                for source in select.select([channel,sock],[],[],30)[0]:
                    data=source.recv(131072)
                    if not data:return
                    (sock if source is channel else channel).sendall(data)
        except (OSError,EOFError):pass
        finally:
            channel.close()
            if sock:sock.close()
    def accept():
        while transport.is_active():
            channel=transport.accept(1)
            if channel:threading.Thread(target=bridge,args=(channel,),daemon=True).start()
    threading.Thread(target=accept,daemon=True).start()
    command=f'cd /root/autodl-tmp/lerobot-main && HTTPS_PROXY=http://127.0.0.1:{port} HTTP_PROXY=http://127.0.0.1:{port} HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_VERBOSITY=error /root/miniconda3/bin/python -u scripts/p47_publish.py'
    stdin,stdout,stderr=client.exec_command(command,timeout=7200)
    stdin.write(os.environ['HF_TOKEN']+'\n');stdin.flush();stdin.channel.shutdown_write()
    channel=stdout.channel;last=time.time()
    try:
        while not channel.exit_status_ready():
            for read,ready in [(channel.recv,channel.recv_ready),(channel.recv_stderr,channel.recv_stderr_ready)]:
                if ready():print(read(32768).decode(errors='replace').split('https://')[0],end='',flush=True)
            if time.time()-last>50:print('UPLOAD_SESSION_ACTIVE',flush=True);last=time.time()
            time.sleep(.2)
        for read,ready in [(channel.recv,channel.recv_ready),(channel.recv_stderr,channel.recv_stderr_ready)]:
            while ready():print(read(32768).decode(errors='replace').split('https://')[0],end='',flush=True)
        code=channel.recv_exit_status();print('PUBLISH_EXIT',code,flush=True)
        if code:raise SystemExit(code)
    finally:
        transport.cancel_port_forward('127.0.0.1',port);client.close()


if __name__=='__main__':main()
