"""Read-only P4.5 queue status; no torch import and no robot I/O."""
import json
from pathlib import Path
import time

ROOT=Path('/root/autodl-tmp/lerobot-main/experiments/p4_final_p1_training/p4_5_robot_reliability_server10h/index1/result/server')


def main():
    campaign=json.loads((ROOT/'campaign_started.json').read_text())
    status=json.loads((ROOT/'controller_status.json').read_text())
    result=dict(controller_pid=campaign['pid'],elapsed_h=(time.time()-campaign['started'])/3600,
        hours_until_hard_stop=max(0,(campaign['hard_end']-time.time())/3600),active=status.get('active',{}),runs={})
    for name in ['VACT-S1','VACT-S2','VACT-S3','NG-L-G','NG-F-G','NG-L-L','NG-F-L']:
        folder=ROOT/'runs'/name;progress=folder/'progress.json';complete=folder/'execution_complete.json'
        result['runs'][name]=json.loads(complete.read_text()) if complete.exists() else json.loads(progress.read_text()) if progress.exists() else dict(status='QUEUED')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
