"""Asset identity checks, frozen source snapshots and unseen LIBERO init registration."""
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from p49_runtime import ROOT, WORK, OUT, R5, P41, FIX, sha, write

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data=ROOT/'outputs/p4_0/formal6_merged'
    revision='83a829af6df245ed53596dbc22b18c19f8bd3315'
    tree=json.loads((data/f'.cache/huggingface/trees/{revision}.json').read_text())['files']
    checks=[]
    for name,item in tree.items():
        p=data/name
        assert p.stat().st_size==item['size'],name
        if 'lfs_sha256' in item:
            actual=sha(p);assert actual==item['lfs_sha256'],name
        else:
            content=p.read_bytes();actual=hashlib.sha1(f'blob {len(content)}\0'.encode()+content).hexdigest()
            assert actual==item['blob_id'],name
        checks.append(dict(path=str(p),hash=actual))
    prior=ROOT/'experiments/p4_final_p1_training/p4_0_1_training_readiness_unblock/index1/result'
    splits=json.loads((prior/'split_freeze_manifest.json').read_text())
    for name,h in splits['episode_file_sha256'].items():assert sha(prior/name)==h
    train=json.loads((prior/'train90_episode_ids.json').read_text())
    dev=json.loads((prior/'dev15_episode_ids.json').read_text())
    final=json.loads((prior/'final15_episode_ids.json').read_text())
    assert len(train)==90 and len(dev)==len(final)==15 and not(set(train)&set(dev) or set(train)&set(final) or set(dev)&set(final))
    legacy_expected=json.loads((P41/'source_hashes.json').read_text())
    for relative in ['src/lerobot/policies/act/modeling_act.py','src/lerobot/policies/act/processor_act.py',
                     'src/lerobot/policies/act/object_conditioning/modeling_object_conditioning.py']:
        assert sha(ROOT/relative)==legacy_expected[relative],relative
    fixed_expected=json.loads((FIX/'result/isolation_manifest.json').read_text())
    assert sha(FIX/'code/src/lerobot/policies/act/modeling_act.py')==fixed_expected['isolated_fixed_sha256']
    for kind,source in [('legacy',ROOT/'src'),('fixed',FIX/'code/src')]:
        dest=WORK/f'workspace/f6_{kind}/src'
        if not dest.exists():shutil.copytree(source,dest,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    helpers=WORK/'workspace/f6_helpers';helpers.mkdir(exist_ok=True)
    for name in ['p4_1_pilot.py','p4_2_queue.py','p4_2_1_repair.py']:
        shutil.copyfile(FIX/'code/scripts'/name,helpers/name)
    r5_assets=json.loads((R5/'result/artifact_manifest.json').read_text())
    for entry in r5_assets.values():assert sha(entry['server_path'])==entry['sha256'],entry['server_path']
    files=[ROOT/'src/lerobot/policies/act/modeling_act.py',FIX/'code/src/lerobot/policies/act/modeling_act.py',
           P41/'formal6_train90_prototypes.pt',P41/'train90_statistics.pt',R5/'result/prototype_bank.pt',R5/'result/train35_statistics.pt']
    write(OUT/'preflight_manifest.json',dict(status='ASSET_IDENTITY_PASS',formal6_revision=revision,
        formal6_cached_pinned_tree_file_checks=checks,split=[len(train),len(dev),len(final)],
        final_access='ID list/hash only; no Final episode decoding, inference, or metrics',
        r5_assets_unchanged=len(r5_assets),assets=[dict(path=str(p),sha256=sha(p)) for p in files],
        initial_formal6_identity='RECIPE_MATCHED_NOT_BYTE_MATCHED; historical step0 not persisted',
        resource=subprocess.check_output(['nvidia-smi'],text=True),free_disk_bytes=shutil.disk_usage(ROOT).free,
        active_processes=subprocess.check_output(['ps','-eo','pid,args'],text=True)))
    # Do not instantiate/reset an environment; inspect the official legal state bank only.
    import sys
    sys.path.insert(0,str(R5/'workspace/scripts'))
    import p48_r5_common
    os.environ['LIBERO_CONFIG_PATH']=str(p48_r5_common.OLD/'libero_config')
    from libero.libero import benchmark
    suite=benchmark.get_benchmark_dict()['libero_object']()
    legal=len(suite.get_task_init_states(0))
    assert legal>=30
    contract=dict(reused=list(range(10,20)),new=list(range(20,30)),seed=1,
        legal_init_state_count=legal,new_list_frozen_before_new_policy_training=True,
        classification='10 reused exploratory + 10 previously unused paired starts; single-task pilot',
        source_r5_timing_sha256=sha(R5/'result/timing_contract.json'))
    write(OUT/'libero_init_contract.json',contract)
    write(OUT/'preflight_complete.json',dict(status='PASS'))

if __name__=='__main__':main()
