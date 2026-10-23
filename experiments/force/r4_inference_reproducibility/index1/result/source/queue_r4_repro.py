"""Sequential R4 gate queue; SSH-independent, stops on failed prerequisites."""
import json
import shutil
import subprocess
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np

from run_r4_repro import WORK, OUT, F5, compare, configure, save, table, digest

MODE='benchmark_off'


def run(stage,seed,tag=None):
    assert shutil.disk_usage(WORK).free>15*1024**3,'DISK_RESERVE_GUARD'
    label=f'S{seed}_{MODE}_{stage}'+('_'+tag if tag else '')
    save(OUT/'queue_state.json',dict(status='RUNNING',current=label,updated_unix=time.time()))
    args=['bash',str(WORK/'launch_r4.sh'),'--stage',stage,'--seed',str(seed),'--mode',MODE]
    if tag:args+=['--tag',tag]
    with (WORK/'logs'/f'{label}.log').open('x') as stream:
        subprocess.run(args,stdout=stream,stderr=subprocess.STDOUT,check=True)


def repeat_comparison(seed):
    root=WORK/'raw'/MODE/f'S{seed}'
    manifests=[json.loads((root/tag/'input_manifest.json').read_text()) for tag in ['D0','D1']]
    assert manifests[0]==manifests[1],'D_MODEL_INPUTS_DIFFER'
    statistics=[]
    for path in sorted((root/'D0/chunks').glob('*.npz')):
        with np.load(path) as a,np.load(root/'D1/chunks'/path.name) as b:
            statistics.append(dict(seed=seed,batch=path.stem,**compare(a['native'],b['native'])))
    table(root/'full_replay_per_batch.csv',statistics)
    with np.load(root/'D0/replay.npz') as a,np.load(root/'D1/replay.npz') as b:
        for key in ['episode','frame','state','raw_load','target_native']:np.testing.assert_array_equal(a[key],b[key])
        h0=float(np.abs(a['prediction_native']-b['prediction_native']).max())
        # Every per-episode aggregate has identical underlying predictions when bitwise equal.
        metric_difference=float(np.max(np.abs(np.abs(a['prediction_native']-a['target_native']).mean(0)-
            np.abs(b['prediction_native']-b['target_native']).mean(0))))
    result=dict(seed=seed,frames=28739,batches=len(statistics),inputs_bitwise_equal=True,
        max_native=max(r['max_native'] for r in statistics),h0_max_native=h0,
        max_joint_mean_absolute_error_difference=metric_difference,passed=all(r['passed'] for r in statistics))
    save(root/'full_replay_comparison.json',result)
    assert result['passed'],'D_REPRODUCIBILITY_FAILED'
    return result


def postfix():
    configure(MODE)
    # An isolated view redirects the inherited analysis; old caches/reports are untouched.
    view=WORK/'postfix_view';view.mkdir(exist_ok=False);(view/'cache').mkdir()
    for name in ['inputs','runs']:(view/name).symlink_to(F5/name,target_is_directory=True)
    for seed in [36,1,2,3,4]:
        label='G36-S3' if seed==36 else f'G60-S{seed}'
        tag='D0' if seed in [1,3] else 'POSTFIX'
        (view/'cache'/f'{label}.npz').symlink_to(WORK/'raw'/MODE/f'S{seed}'/tag/'replay.npz')
    destination=OUT/'postfix_f5_1';destination.mkdir()
    shutil.copy2(F5/'result/full60_manifest.csv',destination/'full60_manifest.csv')
    import run_f5_1_r3 as worker
    import report_f5_1_r3 as report
    worker.WORK=view;report.WORK=view;report.OUT=destination
    report.core_or_full(final=False)
    # Reuse the same normalized metrics, grouping and classification definitions.
    shutil.copy2(destination/'unified_full60_replay.csv',OUT/'postfix_f5_1_metrics.csv')
    save(OUT/'postfix_provenance.json',dict(evaluation_mode=MODE,training_changes=False,
        original_report_script_sha256=digest(F5/'report_f5_1_r3.py'),
        original_analysis_sha256=digest(F5/'f5_1_analysis.py'),
        note='Fresh globals plus frozen local weights; no pre-fix predictions reused; analysis view symlinks read-only model/data sources'))


def main():
    try:
        deadline=time.monotonic()+1200
        first=WORK/'raw'/MODE/'S3/D0/complete.json'
        while not first.exists():
            if time.monotonic()>deadline:raise RuntimeError('EXISTING_D0_NOT_COMPLETED_WITHIN_20MIN; inspect retained log')
            time.sleep(10)
        run('full',3,'D1')
        results=[repeat_comparison(3)]
        for stage in ['probe','cross','preprocess']:run(stage,1)
        for tag in ['D0','D1']:run('full',1,tag)
        results.append(repeat_comparison(1))
        table(OUT/'full_replay_repeat.csv',results)
        table(OUT/'seed1_repeat_verification.csv',[results[1]])
        save(OUT/'repro_gate.json',dict(R4_REPRODUCIBILITY_PASS=True,mode=MODE,results=results,
            action_limit_exclusive=.001,training_changed=False,robot_authorized=False))
        for seed in [36,2,4]:run('full',seed,'POSTFIX')
        postfix()
        save(OUT/'queue_state.json',dict(status='COMPUTE_DONE_REQUIRES_REVIEW_AND_RESULT',
            R4_COMPLETE=False,R4_REPRODUCIBILITY_PASS=True,updated_unix=time.time()))
    except Exception:
        error=traceback.format_exc();print(error,flush=True)
        save(OUT/'queue_state.json',dict(status='STOPPED_ERROR_RETAINED',R4_COMPLETE=False,
            error=error,updated_unix=time.time()))
        raise
    finally:
        save(OUT/'evidence_manifest.json',dict(server_root=str(WORK),sha256={str(p.relative_to(WORK)):digest(p)
            for p in WORK.rglob('*') if p.is_file() and p.suffix in ['.py','.sh','.json','.csv','.md'] and
            'postfix_view' not in p.parts and p.name!='evidence_manifest.json'}))
        with zipfile.ZipFile(WORK/'R4_partial_result.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for path in OUT.rglob('*'):
                if path.is_file():archive.write(path,'result/'+path.relative_to(OUT).as_posix())


if __name__=='__main__':main()
