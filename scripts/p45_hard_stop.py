"""Independent hard deadline for this campaign's exact recorded worker process groups."""
import csv
import json
import os
from pathlib import Path
import signal
import time

OUT=Path('/root/autodl-tmp/lerobot-main/experiments/p4_final_p1_training/p4_5_robot_reliability_server10h/index1/result/server')


def alive(pid,name):
    path=Path(f'/proc/{pid}/cmdline')
    try:return path.exists() and b'p45_' in path.read_bytes() and ('--run\0'+name+'\0').encode() in path.read_bytes()
    except FileNotFoundError:return False


def main():
    deadline=json.loads((OUT/'campaign_started.json').read_text())['hard_end']
    while time.time()<deadline:time.sleep(max(0,min(30,deadline-time.time())))
    with (OUT/'schedule_actual.csv').open() as stream:rows=list(csv.DictReader(stream))
    targets={(int(row['pid']),row['run']) for row in rows if row['event']=='START'}
    events=[]
    for sig in [signal.SIGTERM,signal.SIGKILL]:
        for pid,name in targets:
            if alive(pid,name):
                try:os.killpg(pid,sig);events.append(dict(pid=pid,run=name,signal=int(sig),time=time.time()))
                except ProcessLookupError:pass
        if sig==signal.SIGTERM:time.sleep(5)
    (OUT/'hard_deadline_events.json').write_text(json.dumps(events,indent=2)+'\n')


if __name__=='__main__':main()
