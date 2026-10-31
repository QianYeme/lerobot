"""Build an honest complete/partial package from actual P4.9 artifacts."""
import csv
import json
from pathlib import Path
import zipfile

from p49_runtime import OUT, WORK, JOBS, sha, write

def merge(paths,dest):
    rows=[]
    for p in paths:
        with p.open() as f:
            for record in csv.DictReader(f):
                parts=p.relative_to(OUT).parts
                record['source_path']=p.relative_to(OUT).as_posix()
                if parts[0]=='runs':record.setdefault('job',parts[1])
                if parts[0]=='libero_rollout_jobs':record.update(model=parts[1],cohort=parts[2])
                rows.append(record)
    if rows:
        fields=list(dict.fromkeys(k for r in rows for k in r))
        with dest.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    return len(rows)

def main():
    registry=json.loads((OUT/'job_registry.json').read_text())
    counts={}
    counts['checkpoints']=merge(sorted((OUT/'runs').glob('*/checkpoint_manifest.csv')),OUT/'checkpoint_manifest.csv')
    provenance={}
    for domain,name in [('f6','formal6'),('lb','libero')]:
        jobs=[j for j,v in JOBS.items() if v[0]==domain]
        counts[name+'_train']=merge([OUT/'runs'/j/'training_metrics.csv' for j in jobs if (OUT/'runs'/j/'training_metrics.csv').exists()],OUT/f'{name}_train_metrics.csv')
        paths=[p for p in (OUT/'evaluation').glob('*/dev_metrics.csv') if any(p.parent.name.startswith(j) for j in jobs)]
        if domain=='lb' and (OUT/'evaluation/ACT_GATE_NA/dev_metrics.csv').exists():paths.append(OUT/'evaluation/ACT_GATE_NA/dev_metrics.csv')
        counts[name+'_dev']=merge(paths,OUT/f'{name}_dev{15 if domain=='f6' else 9}_paired.csv')
        object_rows=[]
        for p in (OUT/'evaluation').glob('*/object_path.json'):
            if any(p.parent.name.startswith(j) for j in jobs):object_rows.extend(json.loads(p.read_text()))
        if object_rows:
            with (OUT/f'{name}_object_path.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(object_rows[0]));w.writeheader();w.writerows(object_rows)
    counts['rollout_rows']=merge(sorted((OUT/'libero_rollout_jobs').glob('*/*/rollouts/*/libero_rollouts.csv')),OUT/'libero_rollouts_per_init.csv')
    for job in JOBS:
        folder=OUT/'runs'/job
        provenance[job]=json.loads((folder/'identity.json').read_text()) if (folder/'identity.json').exists() else {'status':registry['jobs'][job]}
    write(OUT/'model_provenance.json',provenance)
    complete=all(v.get('status')=='DONE' for v in registry['jobs'].values()) and registry.get('all_evaluation_tasks_done',False)
    state=dict(status='RESULT_READY' if complete else 'PARTIAL_OR_BLOCKED',counts=counts,
        jobs=registry['jobs'],independent_builder_review='PENDING',independent_redteam_review='PENDING',
        real_robot='NOT_RUN',final_data='NOT_EVALUATED',published=False)
    write(OUT/'execution_complete.json',state)
    (OUT/'robot_candidate_handoff.md').write_text('''# Robot candidate handoff

No real robot was operated. Do not infer physical performance from teacher-forced MAE.
Historical L1 (Legacy seed1) and F3 (Fixed seed3) remain controls. Each new run changes only gate initialization to +0.001; the scalar remains learnable and signed.
Use checkpoint_manifest.csv and model_provenance.json for exact steps, paths and hashes. Only a complete80k formal6 model is eligible for later controlled robot review; PARTIAL is not80k.
Legacy uses the original normalized Top as DINO input (the frozen encoder normalizes it again). Fixed preserves Top [0,1] under observation.images.top_raw for one DINO normalization. Main CNN preprocessing is unchanged. Never interchange these entry paths.
State8 includes six joints plus gripper.load/gripper.curr; action6, chunk100, historical train-only normalizer/prototype bank. Source interface is640x480; nn SO101 calibration/units retain FRESH_TRAIN_LIMITED history and require on-site review. Camera/port history: follower /dev/ttyACM0, leader /dev/ttyACM1, gripper2/top4. This is not authorization to send actions or a new deployment command.
Future pairings: L1 Gate0 vs L1 Gate+ seed1; F3 Gate0 vs F3 Gate+ seed3. Cross-seed Legacy-vs-Fixed differences are confounded. No automatic HF upload, robot test, or next experiment.
''')
    (OUT/'architecture_diff_and_init_contract.md').write_text('''# Initialization contract

Only scientific variable: learnable scalar gate initial0 -> +.001. Signed scalar, adapter, attention, tokens, residual scale, data, seeds, loss and budgets remain historical. Local run directories/source snapshots are administrative changes. Machine identity and zero-vs-plus non-gate tensor equality are recorded per smoke job.
LIBERO full reconstructed initial hash must match frozen r5; formal6 historical original step0 was not persisted: RECIPE_MATCHED_NOT_BYTE_MATCHED, even though reconstructed zero/plus non-gate tensors match. No old trained weights initialize new runs. Historical trained checkpoints are strictly loaded only for baseline checks/scoring.
Each pair uses identical evaluation code/observations. Dev15 uses the historical six fractions per episode (90 observations), not all Dev frames; Dev9 uses every1391 frame. Final episode IDs are inspected only for split disjointness; no Final decoding or inference. File container hashing for revision verification is not episode access.
Temporary zero-gate evaluations measure mechanism only. New/reference alpha remains unchanged after probes. Raw units differ across formal6 and LIBERO; do not pool MAE across domains.
''')
    (OUT/'builder_review.md').write_text('# Builder review\n\nIndependent: PENDING. See preflight_manifest, per-job smoke validation, resume hashes, benchmark final-weight hashes, source snapshots, actual checkpoint manifests and execution_complete. Missing items must remain missing, not PASS by assumption.\n')
    (OUT/'red_team_review.md').write_text('# Red-Team review\n\nIndependent: PENDING. Main threats: formal6 cross-preprocessing seed confounding; historical step0 unavailable; tiny single-task init sample; r5 starts reused exploratory; manually assisted four Train reference points; offline MAE is not grasp success; Fixed preprocessing must not be used for Legacy; no Final episode decoding; no task expansion. Simulator timebase remains native one action per20Hz step with10FPS encoding provenance inherited, not interpolated or asserted equivalent physical seconds.\n')
    (OUT/'unresolved_proposals.md').write_text('# Unresolved / review\n\n'+json.dumps(registry.get('failures',[]),indent=2)+'\n\nAll pair-level benefit labels remain INCONCLUSIVE pending evidence review; formal6 has no new real-robot evidence. Do not automatically change LR/bank/seed/architecture/training budget or start additional experiments.\n')
    (OUT/'summary.md').write_text('# P4.9-r1 result\n\n'+json.dumps(state,indent=2)+'\n\nRead measured CSVs and per-job raw logs. Four jobs and all evaluation dependencies must finish before RESULT_READY. No success, causal superiority or deployment readiness is inferred from training completion.\n')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(10,5))
        for job in JOBS:
            path=OUT/'runs'/job/'training_metrics.csv'
            if path.exists():
                with path.open() as f:rows=list(csv.DictReader(f))
                ax.plot([int(r['step']) for r in rows],[float(r['gate']) for r in rows],label=job)
        ax.set_xlabel('effective optimizer step');ax.set_ylabel('learned signed gate');ax.legend();fig.tight_layout()
        fig.savefig(OUT/'gate_trajectories.png');plt.close(fig)
    except ImportError:pass
    artifacts=[p for p in OUT.rglob('*') if p.is_file() and p.suffix not in ['.pt','.pth','.safetensors','.lock'] and p.name!='artifact_manifest.json']
    manifest={p.relative_to(WORK).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in artifacts}
    write(OUT/'artifact_manifest.json',manifest)
    with zipfile.ZipFile(WORK/'P4_9_r1_result.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in artifacts+[OUT/'artifact_manifest.json',WORK/'decision.md']:z.write(p,p.relative_to(WORK).as_posix())
    with zipfile.ZipFile(WORK/'P4_9_r1_result.zip') as z:assert z.testzip() is None
    print(json.dumps(state),flush=True)

if __name__=='__main__':main()
