"""Explain frozen G2 rejections and render post-selection evidence, without retuning."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path('/root/autodl-tmp/lerobot-main')
OUT = ROOT/'experiments/p4_public_benchmark/p4_8_libero_pilot/index3/result'
OLD = OUT.parent.parent/'index1/result'
DATA = Path('/root/autodl-tmp/p48_data/lerobot_libero')
sys.path.insert(0, str(OLD/'workspace/lerobot/src'))


def write(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    import av
    import cv2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    torch.set_num_threads(4)
    cache = torch.load(OUT/'train_only_feature_track_cache.pt', map_location='cpu', weights_only=False)
    rows = []
    for r in cache:
        p = r['proxy']
        masks = [('pre_static',p['pre']<=1.5),('post_moves',p['post']>=2.),
                 ('feature_stable',p['stability']>=.65),('flow_cycle',p['cycle_error']<=2.5),
                 ('ee_speed_corr',p['correlation']>=.1),('near_premover',p['proximity']<=42.)]
        cumulative = np.ones(256,dtype=bool)
        for name, mask in masks:
            cumulative &= mask
            rows.append(dict(episode=r['episode'],audit_only=r['audit_only'],criterion=name,
                             standalone_pass=int(mask.sum()),cumulative_pass=int(cumulative.sum())))
        assert int(p['positive'].sum()) <= int(cumulative.sum())
    with (OUT/'candidate_filter_counts.csv').open('w') as f:
        w = csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    figures = OUT/'figures';figures.mkdir(exist_ok=True)
    fig,axes = plt.subplots(1,2,figsize=(13,4),layout='constrained')
    counts = [int(r['proxy']['positive'].sum()) for r in cache]
    axes[0].bar(range(len(cache)),counts,color=['#dd9c43' if r['audit_only'] else '#426f99' for r in cache])
    axes[0].set(xlabel='Frozen Train35 order (orange: reserved proxy audit)',ylabel='Eligible patches',title='Automatic candidates by episode')
    axes[1].bar(['Discovery','Reserved audit'],[sum(c for c,r in zip(counts,cache) if not r['audit_only']),sum(c for c,r in zip(counts,cache) if r['audit_only'])],color=['#426f99','#dd9c43'])
    axes[1].axhline(16,color='#ac3333',linestyle='--',label='Pre-registered discovery minimum 16')
    axes[1].legend();axes[1].set(ylabel='Candidate count',title='Coverage gate; proxies are not target labels')
    fig.savefig(figures/'automatic_candidate_coverage.png',dpi=180);plt.close(fig)
    provenance = json.loads((OUT/'prototype_provenance.json').read_text())
    meta = pd.read_parquet(DATA/'meta/episodes/chunk-000/file-000.parquet')
    from PIL import Image,ImageDraw
    sheet = Image.new('RGB',(9*224,4*250),(255,255,255));draw = ImageDraw.Draw(sheet)
    for rank, source in enumerate(provenance):
        r = next(x for x in cache if x['episode']==source['episode']);m = meta.loc[meta.episode_index == r['episode']].iloc[0]
        key = 'videos/observation.images.image/'
        file = DATA/'videos/observation.images.image'/f'chunk-{int(m[key+"chunk_index"]):03d}'/f'file-{int(m[key+"file_index"]):03d}.mp4'
        targets = {round(float(m[key+'from_timestamp'])*10)+frame:i for i,frame in enumerate(r['frames'])}
        with av.open(str(file)) as stream:
            for index, frame in enumerate(stream.decode(video=0)):
                if index in targets:
                    i = targets[index];rgb = cv2.resize(frame.to_ndarray(format='rgb24')[::-1,::-1],(224,224))
                    image = Image.fromarray(rgb);d = ImageDraw.Draw(image)
                    x,y = r['proxy']['paths'][i,source['patch_index']]
                    d.rectangle((float(x-7),float(y-7),float(x+7),float(y+7)),outline=(255,80,60),width=2)
                    sheet.paste(image,(i*224,rank*250+24))
                    draw.text((i*224+3,rank*250+4),f'ep{r["episode"]} f{r["frames"][i]} AUTO TRACK',fill='black')
                if index >= max(targets):break
    sheet.save(figures/'auto_selected_temporal_tracks.png')

    # Actual historical DINO forward; no policy model or ACT/CNN behavior is claimed here.
    path = ROOT/'src/lerobot/policies/act/object_conditioning/modeling_object_conditioning.py'
    spec = importlib.util.spec_from_file_location('p48_dino_evidence',path);module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    encoder = module.FrozenDINOv2Small('dinov2_vits14').cuda().eval()
    episode = cache[0]
    m = meta.loc[meta.episode_index == episode['episode']].iloc[0];key = 'videos/observation.images.image/'
    file = DATA/'videos/observation.images.image'/f'chunk-{int(m[key+"chunk_index"]):03d}'/f'file-{int(m[key+"file_index"]):03d}.mp4'
    target = round(float(m[key+'from_timestamp'])*10)+episode['frames'][3]
    with av.open(str(file)) as stream:
        rgb = next(frame.to_ndarray(format='rgb24') for i,frame in enumerate(stream.decode(video=0)) if i==target)
    rgb = cv2.resize(rgb[::-1,::-1],(224,224))
    raw = torch.from_numpy(rgb.copy()).permute(2,0,1).float()[None].cuda()/255
    mean = raw.new_tensor([.485,.456,.406]).view(1,3,1,1);std = raw.new_tensor([.229,.224,.225]).view(1,3,1,1)
    capture = [];original = encoder.model.forward_features
    def observed(x):
        capture.append(x.detach().cpu());return original(x)
    encoder.model.forward_features = observed
    with torch.inference_mode():
        fixed,_ = encoder(raw);legacy,_ = encoder((raw-mean)/std);prototype,_ = encoder(raw)
    assert torch.equal(capture[0],capture[2]) and not torch.equal(capture[0],capture[1])
    assert torch.equal(fixed,prototype) and not torch.equal(fixed,legacy)
    write('legacy_fixed_preprocess_probe.json',dict(status='ACTUAL_FROZEN_DINO_FORWARD_PASS',
          episode=episode['episode'],frame=episode['frames'][3],train_only=True,
          source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
          resize='RGB uint8 OpenCV bilinear to224, matching frozen prototype builder; online policy integration NOT_RUN',
          input_shape=list(capture[0].shape),feature_shape=list(fixed.shape),
          fixed_prototype_input_max_diff=float((capture[0]-capture[2]).abs().max()),
          legacy_fixed_input_max_diff=float((capture[0]-capture[1]).abs().max()),
          legacy_fixed_feature_mean_abs_diff=float((fixed-legacy).abs().mean()),
          fixed_prototype_feature_max_diff=float((fixed-prototype).abs().max()),
          p1_policy_forward_executed=False,cnn_invariance_in_full_p1='NOT_RUN',formal_updates=0))
    print('GATE_FAILURE_EVIDENCE_AND_ACTUAL_DINO_PREPROCESS_PROBE_COMPLETE',flush=True)


if __name__ == '__main__':main()
