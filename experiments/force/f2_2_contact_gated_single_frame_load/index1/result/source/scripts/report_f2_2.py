"""Audit completed F2.2 evidence and create the bounded handoff package."""
import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def table(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def validate_curve(directory, dev):
    records = []
    for step in [20000, 40000, 60000, 80000, 100000]:
        score = read(directory/'evaluation'/f'{step:06d}.json')
        assert score['frames'] == 5748
        assert set(map(int, score['per_episode'])) == set(dev)
        weighted = sum(r['primary']*r['frames'] for r in score['per_episode'].values()) / 5748
        assert abs(weighted-score['primary']) < 1e-8
        assert all(np.isfinite(score[k]) for k in ['primary', 'physical_gripper_mae', 'physical_arm_mae'])
        records.append(dict(step=step, primary=score['primary'], arm_mae=score['physical_arm_mae']))
    expected = min(records, key=lambda r:(r['primary'], r['step']))
    selected = read(directory/'selected.json')
    assert selected['step'] == expected['step'] and selected['primary'] == expected['primary']
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    base = args.base.resolve()
    out = base/'delivery'
    spec = read(base/'inputs/split_manifest.json')
    gate = read(base/'inputs/gate_config.json')
    protocol = read(base/'inputs/protocol.json')
    evaluation = read(out/'selected_evaluation.json')
    costs = read(out/'latency_and_params.json')
    assert hashlib.sha256((base/'inputs/gate_config.json').read_bytes()).hexdigest() == protocol['gate_sha256']
    assert read(base/'results/smoke.json')['full_initialization_equal']
    assert not set(spec['train']) & set(spec['dev'])
    assert len(spec['train']) == 36 and len(spec['dev']) == 12
    health, curves, manifest, episode_rows, cf_rows, identity = {}, {}, [], [], [], {}
    for variant in ['B0', 'B1']:
        directory = base/'results'/variant
        assert (base/f'{variant}.exit').read_text().strip() == '0'
        assert read(directory/'complete.json')['steps'] == 100000
        curves[variant] = validate_curve(directory, spec['dev'])
        with (directory/'training.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        assert [int(r['step']) for r in rows] == [1] + list(range(100, 100001, 100))
        assert all(np.isfinite(float(v)) for r in rows for v in r.values())
        assert all(float(r['elapsed_s']) >= 0 for r in rows)
        health[variant] = dict(rows=len(rows), last_step=100000, exit_code=0,
            max_logged_grad_norm=max(float(r['grad_norm']) for r in rows),
            finite_logged_values=True, checkpoints=5, caveat='100-step logs do not expose every batch; runner rejects nonfinite loss/norm every step')
        info = evaluation[variant]
        selected = info['selection']
        checkpoint = Path(selected['checkpoint'])
        identity[variant] = dict(checkpoint=str(checkpoint), step=selected['step'],
            model_sha256=costs['variants'][variant]['model_sha256'], config=read(checkpoint/'config.json'))
        for ep, metrics in info['metrics']['per_episode'].items():
            episode_rows.append(dict(variant=variant, episode=int(ep), **metrics))
        for mode, values in info['counterfactuals'].items():
            cf_rows.append(dict(variant=variant, mode=mode, frames=values['frames'],
                normalized_gripper_delta=values['normalized_gripper_delta'], normalized_arm_delta=values['normalized_arm_delta'],
                physical_gripper_delta=values['physical_gripper_delta'], physical_arm_delta=values['physical_arm_delta'],
                counterfactual_primary=values['counterfactual_primary']))
        manifest.append(dict(variant=variant, seed=1, steps=100000, batch_size=4,
            selected_step=selected['step'], selected_checkpoint=str(checkpoint),
            log=str(base/f'{variant}.log'), training_csv=str(directory/'training.csv'), exit_code=0))
    config0, config1 = identity['B0']['config'], identity['B1']['config']
    differences = {key:[config0.get(key), config1.get(key)] for key in set(config0)|set(config1)
                   if config0.get(key) != config1.get(key)}
    assert differences == {'load_feature_gating':[False, True]}, differences
    save(out/'baseline_identity.json', dict(parent='F2.1 A1 architecture, fresh common initialization',
        variants=identity, config_differences=differences, smoke=read(base/'results/smoke.json')))
    save(out/'training_health.json', health)
    table(out/'training_manifest.csv', manifest)
    table(out/'per_episode_metrics.csv', episode_rows)
    table(out/'gate_counterfactuals.csv', cf_rows)
    table(out/'checkpoint_curve.csv', [dict(variant=v, **r) for v in curves for r in curves[v]])
    # Recompute gate activity from canonical per-episode caches. Final arrays are never opened.
    gate_rows, bin_rows, timelines = [], [], {}
    for role in ['train', 'dev']:
        for ep in spec[role]:
            with np.load(base/'inputs/arrays'/f'ep{ep:03d}.npz') as z:
                pos, command = z['state'][:,5], z['action'][:,5]
                active = pos <= gate['threshold']
                gate_rows.append(dict(split=role, episode=ep, frames=len(pos), active=int(active.sum()), fraction=float(active.mean())))
                if role == 'dev':
                    timelines[ep] = active.astype(float)
                for variable, value in [('position', pos), ('recorded_command_descriptive_only', command)]:
                    for low in range(-20, 120, 10):
                        mask = (value >= low) & (value < low+10)
                        bin_rows.append(dict(split=role, episode=ep, variable=variable, low=low, high=low+10,
                                             frames=int(mask.sum()), active=int(active[mask].sum())))
    table(out/'gate_activity.csv', gate_rows)
    table(out/'gate_activity_bins.csv', bin_rows)
    dev_rows = [r for r in gate_rows if r['split']=='dev']
    fraction = sum(r['active'] for r in dev_rows)/sum(r['frames'] for r in dev_rows)
    degenerate = fraction <= .05 or fraction >= .95
    b0, b1 = evaluation['B0']['metrics'], evaluation['B1']['metrics']
    improvements = sum(b1['per_episode'][str(ep)]['primary'] < b0['per_episode'][str(ep)]['primary'] for ep in spec['dev'])
    latency_ratio = costs['variants']['B1']['wall_ms_median']/costs['variants']['B0']['wall_ms_median']
    mechanism = evaluation['B1']['counterfactuals']
    # Descriptive interpretation rubric stored before selected policy results, not a metric change.
    rubric = read(base/'inputs/reporting_rubric.json')
    used = all(mechanism[k]['normalized_gripper_delta'] > rubric['nonzero_diagnostic_floor'] for k in ['gate_off','gate_on'])
    if degenerate:
        outcome = 'GATE_DEGENERATE'
    elif b1['primary'] < b0['primary'] and improvements >= rubric['consistent_episodes_min'] and b1['physical_arm_mae'] <= b0['physical_arm_mae'] and used and latency_ratio <= rubric['latency_ratio_max']:
        outcome = 'CONTACT_GATING_PROMISING'
    elif b1['primary'] > b0['primary'] and sum(b1['per_episode'][str(ep)]['primary'] > b0['per_episode'][str(ep)]['primary'] for ep in spec['dev']) >= rubric['consistent_episodes_min']:
        outcome = 'GATING_HURTS'
    elif b1['primary'] >= b0['primary']:
        outcome = 'ALWAYS_ON_LOAD_SUFFICIENT'
    else:
        outcome = 'UNRESOLVED'
    metrics = dict(outcome=outcome, evidence='SINGLE_SEED_EXPLORATORY', variants=evaluation,
        relative_primary_change_b1_vs_b0=b1['primary']/b0['primary']-1,
        episodes_improved=improvements, gate_fraction=fraction, gate_degenerate=degenerate,
        arm_mae_ratio=b1['physical_arm_mae']/b0['physical_arm_mae'], latency_ratio=latency_ratio,
        rubric=rubric, independent_builder_red_team_review='PENDING')
    save(out/'metrics.json', metrics)
    figures = out/'figures'; figures.mkdir(exist_ok=True)

    def finish(name, ylabel, title):
        plt.ylabel(ylabel); plt.title(title); plt.tight_layout()
        plt.savefig(figures/name, dpi=150); plt.close()

    plt.figure(figsize=(6,4)); plt.bar(['B0','B1'],[b0['primary'],b1['primary']])
    finish('b0_b1_primary_metric.png','Normalized h0 gripper L1 (lower better)','Selected Dev12; single seed')
    plt.figure(figsize=(9,4)); xs=np.arange(12)
    for v, offset in [('B0',-.18),('B1',.18)]:
        plt.bar(xs+offset,[evaluation[v]['metrics']['per_episode'][str(ep)]['primary'] for ep in spec['dev']],width=.36,label=v)
    plt.xticks(xs,spec['dev']); plt.xlabel('Dev episode'); plt.legend()
    finish('per_episode_primary_metric.png','Normalized h0 gripper L1','Per-episode primary metric')
    plt.figure(figsize=(10,5))
    plt.imshow(np.stack(list(timelines.values())),aspect='auto',vmin=0,vmax=1,cmap='Blues')
    plt.yticks(range(12),list(timelines)); plt.xlabel('Frame (descriptive only)')
    finish('gate_activity_timeline.png','Dev episode','Frozen gate activity (not contact ground truth)')
    for variable, name in [('position','gate_activity_vs_gripper_position.png'),('recorded_command_descriptive_only','gate_activity_vs_command.png')]:
        plt.figure(figsize=(7,4))
        lows = list(range(-20,120,10)); values=[]
        for low in lows:
            records=[r for r in bin_rows if r['split']=='dev' and r['variable']==variable and r['low']==low]
            count=sum(r['frames'] for r in records)
            values.append(sum(r['active'] for r in records)/count if count else np.nan)
        plt.bar(lows,values,width=8); plt.ylim(0,1.05); plt.xlabel(variable+' (10-unit bins)')
        finish(name,'Active fraction','Command is descriptive, never a gate input')
    plt.figure(figsize=(7,4))
    names=['gate_off','gate_on','feature_zero','input_zero']
    plt.bar(names,[mechanism[n]['normalized_gripper_delta'] for n in names])
    finish('gate_counterfactual_effect.png','Mean absolute normalized h0 change','B1: gate-off = encoded-feature-zero')
    plt.figure(figsize=(6,4)); plt.bar(['B0','B1'],[b0['physical_arm_mae'],b1['physical_arm_mae']])
    finish('arm_mae_comparison.png','Denormalized h0 arm MAE','Mean over five arm joints (lower better)')
    plt.figure(figsize=(9,4))
    for variant in ['B0','B1']:
        with (base/'results'/variant/'training.csv').open() as stream:
            rows=list(csv.DictReader(stream))
        plt.plot([int(r['step']) for r in rows],[float(r['loss']) for r in rows],label=variant,alpha=.7)
    plt.yscale('log'); plt.xlabel('Training step'); plt.legend()
    finish('training_curves.png','Loss (log scale)','Unmodified raw 100-step training logs')
    for path in figures.glob('*.png'):
        pixels=plt.imread(path)
        assert pixels.size > 1000 and np.isfinite(pixels).all() and pixels.std() > .01
    assert len(list(figures.glob('*.png'))) == 8
    for name in ['gate_config.json','protocol.json','normalization.json','split_manifest.json','reporting_rubric.json','input_hashes.json']:
        shutil.copy2(base/'inputs'/name, out/name)
    source_hashes=read(base/'inputs/source_hashes.json')
    for name, expected in source_hashes.items():
        assert hashlib.sha256((base/'source'/name).read_bytes()).hexdigest()==expected, name
    save(out/'implementation_source_hashes.json',source_hashes)
    source_archive=out/'source'; source_archive.mkdir(exist_ok=True)
    for name in source_hashes:
        target=source_archive/name; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(base/'source'/name,target)
    (out/'variant_configs').mkdir(exist_ok=True)
    save(out/'variant_configs/B0.json',config0); save(out/'variant_configs/B1.json',config1)
    for name in ['implementation_patch.diff','decision.md','gpu_smoke.json','historical_metric_reproduction.json']:
        shutil.copy2(base/'inputs'/name,out/name)
    inventory=[]
    evidence=out/'raw_evidence'; evidence.mkdir(exist_ok=True)
    for variant in ['B0','B1']:
        dest=evidence/variant; dest.mkdir(exist_ok=True)
        for name in ['training.csv','selected.json','complete.json']:
            shutil.copy2(base/'results'/variant/name,dest/name)
        shutil.copy2(base/f'{variant}.log',dest/f'{variant}.log')
        shutil.copytree(base/'results'/variant/'evaluation',dest/'evaluation',dirs_exist_ok=True)
        for path in sorted((base/'results'/variant/'checkpoints').rglob('*')):
            if path.is_file():
                inventory.append(dict(path=str(path),bytes=path.stat().st_size,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    save(out/'checkpoint_inventory.json',inventory)
    def doc(name,text):
        (out/name).write_text(text,encoding='utf-8')
    doc('training_health.md','# Training health\n\nBoth variants completed100000 steps with exit0; 1001 finite logged rows each; all five Dev evaluations audited.\n\n'+json.dumps(health,indent=2)+'\nNo automatic restart, skipped bad batch, budget extension or checkpoint deletion.\n')
    doc('implementation_diff.md','# Implementation difference\n\nSelected full configs differ ONLY in load_feature_gating=false/true. Full common initialization equal (gpu_smoke.json). Gate is multiplied after load MLP and does not gate state/image/action or remove positional embedding.\n\nimplementation_patch.diff compares against preserved F2.1 source; hashes pin actual runtime files. Old runner reused unchanged; subclass overrides config, batch and diagnostics. New evaluation never trains.\n')
    doc('leakage_audit.md','# Leakage audit\n\nTrain36 statistics and gate frozen before F2.2 Dev performance. Final12 samples never requested by these runners. Dev used for checkpoint selection and reporting (selection bias disclosed). Episode/frame IDs only index observations; gate uses raw position at t only. Same-frame recorded target is descriptive in plots, not policy/gate input. No GT contact/manual phase labels or progress input.\n\nOriginal dataset state contains load/current, but runner explicitly reconstructs six positions and separate single normalized load; current and cached W15 are not policy inputs. Full source/split/norm/gate hashes preserved. This is an implementation audit, not a system-call proof that physical parquet pages contain no other rows.\n')
    doc('mechanism_review.md','# Mechanism review\n\nFull selected Dev5748 frames per model. B1 gate-off and encoded-feature-zero are the SAME intervention, not independent evidence. Gate-on tests selective suppression. Input-zero means normalized load=0 (Train mean) before biased MLP; separate from feature-zero. Counterfactuals report mean absolute h0 prediction change in normalized and denormalized units, plus counterfactual primary error. Raw predictions and gate in NPZ. No claim of calibrated force or causal contact detection.\n\nGate activity uses position only; can represent a hand-coded grasp regime and trigger in empty closure. Constant token/embedding can still convey state; capacity/regularization confounding remains.\n')
    doc('validation.md','# Completion validation\n\nBoth formal100k budgets, exit0, all five evaluations and earliest-minimum selection checked. Per-episode weighted primary reconstructs stored metric; selected forward reproduced within1e-5. Strict load/config/normalization, finite outputs, gate locality assertions passed. Gate/protocol/runtime hashes verified; exact B0/B1 config difference audited. Eight PNGs passed finite/nonblank validation; independent visual/manual review pending.\n\nLocal causal/boundary pytest and py_compile were preflight checks; no remote pytest claim. Metrics are offline teacher-forced h0, NOT rollout or real-robot performance. No Final, new seed, altered threshold, extra training, deployment, publication or F2.3. Builder/Red-Team independent review pending.\n')
    doc('proposal.md','# Handoff\n\nStop after this package. No next experiment authorized. Outcome: '+outcome+'. Discuss evidence with user/planner; no post-hoc gate tuning, extra seed, residual controller, Final or real-robot rescue.\n\nIf UNRESOLVED: inspect mixed primary/episode/arm/cost/mechanism evidence without modifying this round.\n')
    doc('summary.md',f'''# F2.2 r0 completed — {outcome}

B0 training DONE; B1 training DONE; Dev12 evaluation DONE; Mechanism diagnostics DONE;
Gate audit DONE; Latency/params DONE; Figures DONE; Result package READY.

B0 selected {evaluation['B0']['selection']['step']} steps: normalized h0 gripper L1 {b0['primary']:.8f},
denormalized gripper MAE {b0['physical_gripper_mae']:.6f}, arm MAE {b0['physical_arm_mae']:.6f}.
B1 selected {evaluation['B1']['selection']['step']} steps: normalized h0 gripper L1 {b1['primary']:.8f},
denormalized gripper MAE {b1['physical_gripper_mae']:.6f}, arm MAE {b1['physical_arm_mae']:.6f}.
B1 primary relative change {(b1['primary']/b0['primary']-1)*100:+.2f}%; improves {improvements}/12 episodes.
Dev gate active {fraction*100:.2f}%; threshold unchanged {gate['threshold']}; latency ratio {latency_ratio:.3f}.

Single-seed exploratory offline evidence only. Gate is a position-based grasp proxy, not contact ground truth.
Dev selection/reporting reuse creates optimism; no Final or real robot. No actual force-control effectiveness claim.
Unfavorable evidence is retained. All10 checkpoint weights and logs remain on server, inventory with hashes attached.
ZIP contains reports/raw evaluations/logs/predictions/source, not multi-GB weights or videos.

Execution stops here. Independent Builder Review and Red-Team Review await user/planner.
''')
    status=dict(B0_training='DONE',B1_training='DONE',Dev12_evaluation='DONE',Mechanism_diagnostics='DONE',
        Gate_audit='DONE',Latency_params='DONE',Figures='DONE',Result_package='READY',F2_2_COMPLETE=True,
        outcome=outcome,independent_review='PENDING',completed_utc=datetime.now(timezone.utc).isoformat())
    save(out/'completion.json',status)
    required=['summary.md','validation.md','baseline_identity.json','gate_config.json','variant_configs',
        'training_manifest.csv','training_health.md','metrics.json','per_episode_metrics.csv','gate_activity.csv',
        'gate_counterfactuals.csv','mechanism_review.md','implementation_diff.md','latency_and_params.json',
        'leakage_audit.md','figures','proposal.md']
    assert all((out/name).exists() for name in required)
    package=base/'F2_2_result.zip'
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.rglob('*')):
            if path.is_file(): archive.write(path,Path('result')/path.relative_to(out))
    with zipfile.ZipFile(package) as archive:
        assert archive.testzip() is None
    save(base/'READY.json',dict(**status,package=str(package),package_sha256=hashlib.sha256(package.read_bytes()).hexdigest()))
    print(json.dumps(status),flush=True)


if __name__ == '__main__':
    main()
