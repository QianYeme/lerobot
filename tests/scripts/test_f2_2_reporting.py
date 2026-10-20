import importlib.util
import json
from pathlib import Path

import pytest
import numpy as np

spec = importlib.util.spec_from_file_location('report_f2_2', Path(__file__).parents[2]/'scripts/report_f2_2.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def fixture_curve(root):
    (root/'evaluation').mkdir()
    for step in [20000,40000,60000,80000,100000]:
        value = dict(frames=5748, primary=.2, physical_gripper_mae=3., physical_arm_mae=2.,
                     per_episode={'3':dict(primary=.2,frames=5748)})
        (root/'evaluation'/f'{step:06d}.json').write_text(json.dumps(value))
    (root/'selected.json').write_text(json.dumps(dict(step=20000,primary=.2)))


def test_selection_earliest_tie(tmp_path):
    fixture_curve(tmp_path)
    assert len(report.validate_curve(tmp_path,[3])) == 5


def test_reject_wrong_selection(tmp_path):
    fixture_curve(tmp_path)
    (tmp_path/'selected.json').write_text(json.dumps(dict(step=40000,primary=.2)))
    with pytest.raises(AssertionError):
        report.validate_curve(tmp_path,[3])


def test_reject_wrong_episode_reduction(tmp_path):
    fixture_curve(tmp_path)
    path=tmp_path/'evaluation/020000.json'
    value=json.loads(path.read_text()); value['per_episode']['3']['primary']=.4
    path.write_text(json.dumps(value))
    with pytest.raises(AssertionError):
        report.validate_curve(tmp_path,[3])


def test_full_report_fixture(tmp_path, monkeypatch):
    """Synthetic TEMP data only: exercise report/figures/ZIP without research evidence."""
    inputs=tmp_path/'inputs'; inputs.mkdir()
    delivery=tmp_path/'delivery'; delivery.mkdir()
    (tmp_path/'source').mkdir()
    train=list(range(100,136)); dev=list(range(12))
    def write(path, value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value))
    gate=dict(threshold=44.)
    write(inputs/'gate_config.json',gate)
    import hashlib
    write(inputs/'protocol.json',dict(gate_sha256=hashlib.sha256((inputs/'gate_config.json').read_bytes()).hexdigest()))
    write(inputs/'split_manifest.json',dict(train=train,dev=dev))
    write(inputs/'reporting_rubric.json',dict(consistent_episodes_min=8,nonzero_diagnostic_floor=1e-6,latency_ratio_max=1.2))
    write(inputs/'source_hashes.json',{})
    write(inputs/'input_hashes.json',{})
    write(inputs/'normalization.json',{})
    for name in ['implementation_patch.diff','decision.md','gpu_smoke.json','historical_metric_reproduction.json']:
        (inputs/name).write_text('synthetic fixture only')
    write(tmp_path/'results/smoke.json',dict(full_initialization_equal=True))
    for ep in train+dev:
        (inputs/'arrays').mkdir(exist_ok=True)
        state=np.zeros((479,6)); state[:,5]=np.arange(479)%80
        np.savez(inputs/'arrays'/f'ep{ep:03d}.npz',state=state,action=state)
    evaluation, costs={}, {}
    for variant in ['B0','B1']:
        directory=tmp_path/'results'/variant
        value=.2 if variant=='B0' else .1
        per={str(ep):dict(primary=value,physical_gripper_mae=2.,physical_arm_mae=1.,frames=479) for ep in dev}
        metrics=dict(primary=value,physical_gripper_mae=2.,physical_arm_mae=1.,normalized_all_l1=value,frames=5748,per_episode=per)
        for step in [20000,40000,60000,80000,100000]:
            write(directory/'evaluation'/f'{step:06d}.json',metrics)
            checkpoint=directory/'checkpoints'/f'{step:06d}'/'pretrained_model'
            write(checkpoint/'config.json',dict(load_feature_gating=variant=='B1'))
            (checkpoint/'model.safetensors').write_bytes(b'synthetic fixture')
        selection=dict(step=20000,primary=value,checkpoint=str(directory/'checkpoints/020000/pretrained_model'))
        write(directory/'selected.json',selection); write(directory/'complete.json',dict(steps=100000))
        (tmp_path/f'{variant}.exit').write_text('0')
        (tmp_path/f'{variant}.log').write_text('synthetic fixture')
        with (directory/'training.csv').open('w',newline='') as stream:
            import csv
            writer=csv.DictWriter(stream,fieldnames=['step','loss','grad_norm','elapsed_s'])
            writer.writeheader()
            for step in [1]+list(range(100,100001,100)):
                writer.writerow(dict(step=step,loss=.1,grad_norm=1.,elapsed_s=step))
        cf={name:dict(frames=5748,normalized_gripper_delta=.1,normalized_arm_delta=.1,
            physical_gripper_delta=1.,physical_arm_delta=1.,counterfactual_primary=.3)
            for name in ['feature_zero','input_zero','gate_off','gate_on']}
        evaluation[variant]=dict(selection=selection,metrics=metrics,counterfactuals=cf)
        costs[variant]=dict(model_sha256='fixture',wall_ms_median=10.)
    write(delivery/'selected_evaluation.json',evaluation)
    write(delivery/'latency_and_params.json',dict(variants=costs))
    monkeypatch.setattr('sys.argv',['report_f2_2.py','--base',str(tmp_path)])
    report.main()
    assert json.loads((tmp_path/'READY.json').read_text())['Result_package']=='READY'
    assert len(list((delivery/'figures').glob('*.png')))==8
