"""Formal6-matched four-reference prototype bank, with explicitly disclosed Codex coordinates."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path('/root/autodl-tmp/lerobot-main')
OUT = ROOT/'experiments/p4_public_benchmark/p4_8_libero_pilot/index5/result'
OLD = OUT.parent.parent/'index1/result'
sys.path.insert(0,str(OLD/'workspace/lerobot/src'))


def main():
    import numpy as np
    from PIL import Image,ImageDraw
    import torch
    import torch.nn.functional as F
    torch.manual_seed(1);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    split=json.loads((OUT/'dataset_split_manifest.json').read_text())
    sources=json.loads((OUT/'prototype_sources.json').read_text())
    coords=json.loads((OUT/'prototype_coordinates.json').read_text())
    expected=[sorted(split['train_episode_ids'])[int(35*q)] for q in [0,.25,.5,.75]]
    assert [s['episode'] for s in sources]==expected
    source=ROOT/'src/lerobot/policies/act/object_conditioning/modeling_object_conditioning.py'
    spec=importlib.util.spec_from_file_location('p48_r5_original_oc',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    encoder=module.FrozenDINOv2Small('dinov2_vits14').cuda().eval()
    bank=[];provenance=[];images=[]
    yy,xx=torch.meshgrid(torch.arange(16,device='cuda'),torch.arange(16,device='cuda'),indexing='ij')
    masks=[]
    for s,c in zip(sources,coords):
        assert (s['episode'],s['frame'],s['source_rgb_sha256'])==(c['episode'],c['frame'],c['source_rgb_sha256'])
        rgb=np.array(Image.open(OUT/s['filename']).convert('RGB'))
        assert hashlib.sha256(rgb.tobytes()).hexdigest()==s['source_rgb_sha256']
        image=torch.from_numpy(rgb).permute(2,0,1).float()[None].cuda()/255
        patches,_=encoder(image)
        rx=35*s['width']/640;ry=35*s['height']/480
        mask=((((xx+.5)*s['width']/16-c['x'])/rx).square()+(((yy+.5)*s['height']/16-c['y'])/ry).square())<=1
        assert int(mask.sum())>0,'Do not change radius, frame or coordinates to bypass empty ROI'
        vector=F.normalize(patches[0,mask.flatten()].mean(0),dim=0).cpu()
        bank.append(vector);images.append(image);masks.append(mask)
        indices=mask.flatten().nonzero().flatten().tolist()
        provenance.append(dict(**c,camera_key=s['camera_key'],width=s['width'],height=s['height'],
             rx=rx,ry=ry,selected_patch_indices=indices,selected_patch_count=len(indices),
             source_frame_selection='sorted Train35 q=0,.25,.5,.75; floor(length*.25)',
             prototype_preprocessing='raw RGB0..1, F.interpolate224 bilinear align_corners=False, ImageNet once',
             split='Train35',supervision='Codex-assisted four target-interior points; not unsupervised'))
    first=torch.stack(bank)
    repeat=torch.stack([F.normalize(encoder(image)[0][0,mask.flatten()].mean(0),dim=0).cpu() for image,mask in zip(images,masks)])
    assert torch.equal(first,repeat) and first.shape==(4,384) and torch.isfinite(first).all()
    assert torch.allclose(first.norm(dim=-1),torch.ones(4),atol=1e-6,rtol=0)
    def serialize(tensor):
        buf=io.BytesIO();torch.save(dict(prototypes=tensor,provenance=provenance),buf);return buf.getvalue()
    payload=serialize(first);again=serialize(repeat)
    assert payload==again
    (OUT/'prototype_bank.pt').write_bytes(payload)
    (OUT/'prototype_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    sheet=Image.new('RGB',(512,560),'white');draw=ImageDraw.Draw(sheet)
    for rank,(s,c,p) in enumerate(zip(sources,coords,provenance)):
        image=Image.open(OUT/s['filename']).convert('RGB');d=ImageDraw.Draw(image)
        d.ellipse((c['x']-p['rx'],c['y']-p['ry'],c['x']+p['rx'],c['y']+p['ry']),outline='yellow',width=1)
        for index in p['selected_patch_indices']:
            x=(index%16+.5)*256/16;y=(index//16+.5)*256/16
            d.rectangle((x-8,y-8,x+8,y+8),outline='lime',width=1)
            d.ellipse((x-1,y-1,x+1,y+1),fill='lime')
        d.ellipse((c['x']-2,c['y']-2,c['x']+2,c['y']+2),fill='red')
        image.save(OUT/'prototype_frames'/f'ep{s["episode"]:04d}_roi_overlay.png')
        x=rank%2*256;y=rank//2*280;sheet.paste(image,(x,y+24));draw.text((x+4,y+5),f'ep{s["episode"]} f{s["frame"]} selected {p["selected_patch_count"]}',fill='black')
    sheet.save(OUT/'prototype_roi_contact_sheet.png')
    weights=Path(torch.hub.get_dir())/'checkpoints/dinov2_vits14_pretrain.pth'
    report=dict(status='GEOMETRY_AND_NUMERICS_PASS_AWAITING_VISUAL_ROI_REVIEW',shape=list(first.shape),
                norms=first.norm(dim=-1).tolist(),pairwise_cosine=(first@first.T).tolist(),
                max_repeat_difference=float((first-repeat).abs().max()),
                bank_sha256=hashlib.sha256(payload).hexdigest(),byte_repeat_identical=True,
                selected_patch_counts=[p['selected_patch_count'] for p in provenance],
                geometry='rx=35*256/640=14; ry=35*256/480=18.6666667; patch centers backprojected to256',
                dino_weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                dino_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                dino_repository_revision='UNKNOWN (existing torch.hub source archive); frozen source tree hash below',
                no_dev_or_simulator_target_input=True,manual_user_annotation_required=False)
    repo=Path(torch.hub.get_dir())/'facebookresearch_dinov2_main'
    h=hashlib.sha256()
    for p in sorted(repo.rglob('*.py')):h.update(p.relative_to(repo).as_posix().encode());h.update(p.read_bytes())
    report['dino_source_tree_py_sha256']=h.hexdigest()
    (OUT/'prototype_numeric_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
