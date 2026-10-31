"""Release the two approved formal6 Gate+ deployments; read token only from stdin."""
from pathlib import Path
import json
import shutil
import subprocess
import sys
import time
import uuid
from urllib.parse import quote

import torch
from safetensors.torch import save_file, load_file
import p47_publish as publisher
from p47_stage_models import sha

ROOT=Path('/root/autodl-tmp/lerobot-main')
WORK=ROOT/'experiments/p4_final_p1_training/p4_9_unified_gate_init/index1/release'

def main():
    token=sys.stdin.readline().strip();assert token
    out=WORK/'result/upload';out.mkdir(parents=True,exist_ok=True)
    inventory=[]
    for role,job,slug in [('L1','F6_LEGACY_GATEPLUS_S1','legacy-s1'),('F3','F6_FIXED_GATEPLUS_S3','fixed-s3')]:
        dest=WORK/'staging'/role
        donor=ROOT/f'experiments/p4_final_p1_training/p4_7/index1/download_verify/{role}'
        if not dest.exists():shutil.copytree(donor,dest,ignore=shutil.ignore_patterns('.cache','__pycache__','*.pyc'))
        latest=json.loads((WORK.parent/f'result/runs/{job}/latest.json').read_text())
        source=Path(latest['path']);assert latest['step']==80000 and sha(source)==latest['sha256']
        saved=torch.load(source,map_location='cpu',weights_only=False);assert saved['step']==80000 and saved['job']==job
        state={k:v.detach().cpu().contiguous().clone() for k,v in saved['model'].items()}
        save_file(state,str(dest/'model.safetensors'))
        loaded=load_file(str(dest/'model.safetensors'));assert all(torch.equal(v,loaded[k]) for k,v in state.items())
        config=json.loads((dest/'config.json').read_text());config['object_conditioning_gate_init']=.001
        (dest/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        manifest=json.loads((dest/'manifest.json').read_text())
        manifest.update(role=role,experiment='P4.9-r1',job=job,proposed_repo_id=f'QYyyyyyyy/lerobot-p49-{slug}-gateplus-80k',
            source_checkpoint_path=str(source),source_checkpoint_sha256=sha(source),step=80000,
            seed=1 if role=='L1' else 3,model_sha256=sha(dest/'model.safetensors'),
            tensor_roundtrip_bitwise=True,real_robot_validation='NOT_RUN',historical_identity={'baseline_role':role},
            gate_initial_value=.001,learned_gate=float(state['model.object_conditioning.gate']))
        manifest.pop('repo_id',None)
        (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        shutil.copyfile(WORK.parent/'REAL_ROBOT_TEST_GUIDE.md',dest/'REAL_ROBOT_TEST_GUIDE.md')
        (dest/'README.md').write_text(f'# P4.9 {job}\n\n80k formal6 Gate+ deployment, private release. Baseline runtime role {role} is preserved for compatibility; these are new weights, not the historical baseline.\n\nSee REAL_ROBOT_TEST_GUIDE.md. No new real-robot results exist. State8/action6; original train-only stats and prototypes. Learned gate is exported unchanged, never reset to its initial value.\n')
        subprocess.run([sys.executable,str(ROOT/'scripts/p47_validate_package.py'),'--bundle',str(dest)],check=True)
        help_result=subprocess.run([sys.executable,str(dest/'control_p47.py'),'--bundle',str(dest),'--help'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        assert help_result.returncode==0,help_result.stderr.decode()
        (out/f'{role}_control_smoke.json').write_text(json.dumps(dict(status='PASS',scope='CLI help only, no robot connection')))
        inventory.append(manifest)
        del saved,state,loaded
    (out/'model_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    publisher.WORK=WORK
    def verified_download_retry(*args,**kwargs):
        # Fresh resolve URLs avoid the repeatedly stale proxy/CDN signed redirect.
        from huggingface_hub.utils import get_session
        repo=kwargs['repo_id'];revision=kwargs['revision'];dest=Path(kwargs['local_dir'])
        role=dest.name;stage=WORK/'staging'/role
        checks=json.loads((stage/'checksums.json').read_text())
        checks.append(dict(path='checksums.json',sha256=sha(stage/'checksums.json'),bytes=(stage/'checksums.json').stat().st_size))
        for entry in checks:
            path=dest/entry['path'];path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists() and sha(path)==entry['sha256']:continue
            for attempt in range(5):
                url=f'https://huggingface.co/{repo}/resolve/{revision}/{quote(entry["path"])}?download=true&fresh={uuid.uuid4().hex}'
                temp=path.with_suffix(path.suffix+'.download')
                offset=temp.stat().st_size if temp.exists() else 0
                if offset>=entry['bytes']:
                    if offset==entry['bytes'] and sha(temp)==entry['sha256']:temp.replace(path);break
                    temp.unlink();offset=0
                headers={'Authorization':'Bearer '+token,'Cache-Control':'no-cache'}
                if offset:headers['Range']=f'bytes={offset}-'
                with get_session().stream('GET',url,headers=headers,follow_redirects=True) as response:
                    response.raise_for_status()
                    append=offset and response.status_code==206
                    if append:assert response.headers['Content-Range'].startswith(f'bytes {offset}-')
                    with temp.open('ab' if append else 'wb') as stream:
                        for block in response.iter_bytes():stream.write(block)
                if temp.stat().st_size==entry['bytes'] and sha(temp)==entry['sha256']:
                    temp.replace(path);break
                print('FRESH_DOWNLOAD_RETRY',role,entry['path'],attempt+1,flush=True)
                if attempt==4:raise RuntimeError('Fresh remote download hash mismatch: '+entry['path'])
                time.sleep(2)
        return str(dest)
    publisher.snapshot_download=verified_download_retry
    publisher.main(token)

if __name__=='__main__':main()
