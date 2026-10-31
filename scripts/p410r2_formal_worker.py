"""Matched fresh formal6 20k Gate0/.05 pilots; no final80k authorization."""
from pathlib import Path
import sys
import argparse

ROOT=Path('/root/autodl-tmp/lerobot-main')
P49=ROOT/'experiments/p4_final_p1_training/p4_9_unified_gate_init/index1'
WORK=ROOT/'experiments/p4_public_benchmark/p4_10_gate_sweep/index3'
sys.path.insert(0,str(P49/'workspace/scripts'))
import p49_runtime as runtime

JOBS={'C_L1_G0':('f6','legacy',1,20000),'C_L1_G05':('f6','legacy',1,20000),
      'C_F3_G0':('f6','fixed',3,20000),'C_F3_G05':('f6','fixed',3,20000)}
GATES={j:.05 if j.endswith('G05') else 0. for j in JOBS}
p=argparse.ArgumentParser();p.add_argument('--job',choices=list(JOBS),required=True)
p.add_argument('--mode',choices=['smoke','bench','train'],required=True);p.add_argument('--tag',default='single');a=p.parse_args()
runtime.JOBS.update(JOBS)
source=(P49/'workspace/scripts/p49_worker.py').read_text()
source=source.replace("assert history['step'] == target","assert history['step'] == 80000")
source=source.replace("[('plus', .001), ('zero', 0.)]","[('plus', GATES[job] if GATES[job] else .001), ('zero', 0.)]")
namespace={'__name__':'p410r2_formal6_pilot','__file__':str(P49/'workspace/scripts/p49_worker.py'),'GATES':GATES}
exec(compile(source,'frozen_p49_worker_formal20k','exec'),namespace)
namespace['OUT']=WORK/'result/formal6_pilot'
original=runtime.factory
def factory(job,folder,gate_init=None):return original(job,folder,GATES[job] if gate_init is None else gate_init)
namespace['factory']=factory
sys.argv=[sys.argv[0],'--job',a.job,'--mode',a.mode,'--tag',a.tag]
namespace['main']()
