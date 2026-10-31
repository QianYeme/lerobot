"""Train-only motion-onset DINO prototypes; proxy gates are not semantic labels."""
import hashlib
import io
import json
from pathlib import Path
import sys
import time

ROOT = Path('/root/autodl-tmp/lerobot-main')
WORK = ROOT / 'experiments/p4_public_benchmark/p4_8_libero_pilot/index3'
OUT = WORK / 'result'
DATA = Path('/root/autodl-tmp/p48_data/lerobot_libero')
ALGORITHM = {
    'name': 'TRAIN_INTERACTION_MOTION_ONSET_V1', 'seed': 1,
    'source_split': 'frozen Train35 only; every third sorted Train episode reserved for proxy audit',
    'event': 'first action[6] crossing from <0 to >0.5; command proxy, not physical contact',
    'offsets_frames': [-12, -8, -4, 0, 4, 8, 12, 16, 20],
    'anchor_offset_index': 3, 'resize': [224, 224], 'flip_height_width_once': True,
    'dino': 'frozen dinov2_vits14; single ImageNet normalization',
    'flow': {'method': 'Farneback', 'pyr_scale': .5, 'levels': 3, 'winsize': 21,
             'iterations': 3, 'poly_n': 5, 'poly_sigma': 1.2, 'flags': 0},
    'candidate': {'pre_median_motion_max_px': 1.5, 'post_median_motion_min_px': 2.,
                  'feature_cosine_min': .65, 'forward_backward_error_max_px': 2.5,
                  'ee_speed_correlation_min': .10, 'premover_proximity_max_px': 42.},
    'negative_proxies': {'background_motion_max_px': .5, 'premover_motion_min_px': 4.,
                         'meaning': 'static pixels / already-moving robot-neighborhood proxies, not GT masks'},
    'selection': 'deterministic score-weighted cosine farthest-point, unique source episodes; single DINO patch',
    'score': 'post/(post+2) * exp(-pre/1.5) * max(stability,0) * max(correlation,0) * exp(-proximity/42)',
    'gate': {'candidate_min': 16, 'discovery_episode_min': 4, 'audit_episode_min': 4,
             'selected_episode_min': 4, 'max_pair_cosine': .995,
             'audit_positive_q75_minus_background_q95_min': .02,
             'audit_positive_q75_minus_premover_q95_min': .02,
             'bank_identical_on_full_repeat': True},
    'no_manual_roi': True, 'no_simulator_privileged_data': True,
    'frozen_before_dev_or_rollout': True,
}


def write(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze():
    import yaml
    path = OUT / 'prototype_algorithm.yaml'
    if path.exists():
        assert yaml.safe_load(path.read_text()) == ALGORITHM, 'Do not tune a frozen algorithm'
    else:
        path.write_text(yaml.safe_dump(ALGORITHM, sort_keys=False))


def sample(field, points):
    import cv2
    return cv2.remap(field, points[:, 0].astype('float32').reshape(-1, 1),
                     points[:, 1].astype('float32').reshape(-1, 1),
                     interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).reshape(len(points), -1)


def track(frames, features, states):
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    gray = [cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in frames]
    params = ALGORITHM['flow']
    args = [params[k] for k in ['pyr_scale', 'levels', 'winsize', 'iterations', 'poly_n', 'poly_sigma', 'flags']]
    forward = [cv2.calcOpticalFlowFarneback(a, b, None, *args) for a, b in zip(gray[:-1], gray[1:])]
    backward = [cv2.calcOpticalFlowFarneback(b, a, None, *args) for a, b in zip(gray[:-1], gray[1:])]
    yy, xx = np.meshgrid(np.arange(16)*14+7, np.arange(16)*14+7, indexing='ij')
    points = np.stack([xx, yy], -1).reshape(-1, 2).astype('float32')
    paths = np.empty((9, 256, 2), dtype='float32');paths[3] = points
    for t in range(3, 8):
        paths[t+1] = paths[t] + sample(forward[t], paths[t])
    for t in range(2, -1, -1):
        paths[t] = paths[t+1] + sample(backward[t], paths[t+1])
    cycle = [np.linalg.norm(sample(forward[t], paths[t]) + sample(backward[t], paths[t+1]), axis=1) for t in range(8)]
    cycle = np.median(cycle, axis=0)
    speed = np.linalg.norm(np.diff(paths, axis=0), axis=-1)
    pre = np.median(speed[:3], axis=0);post = np.median(speed[3:], axis=0)
    # DINO patch centers are at 7+14*i; grid_sample uses the corresponding 16x16 grid.
    grid = torch.from_numpy((paths-7)/14/15*2-1).view(9, 1, 256, 2)
    feats = F.normalize(features.float(), dim=-1).reshape(9, 16, 16, 384).permute(0, 3, 1, 2)
    along = F.grid_sample(feats, grid, mode='bilinear', padding_mode='border', align_corners=True).squeeze(2).permute(0, 2, 1)
    along = F.normalize(along, dim=-1)
    stable = (along * along[3:4]).sum(-1).median(0).values.numpy()
    ee = np.linalg.norm(np.diff(states[:, :3], axis=0), axis=-1)
    centered = speed-speed.mean(0, keepdims=True);ec = ee-ee.mean()
    corr = (centered*ec[:, None]).sum(0) / (np.linalg.norm(centered, axis=0)*np.linalg.norm(ec)+1e-8)
    robot = (pre >= 4.) & (post >= 2.)
    distance = np.linalg.norm(points[:, None]-points[robot][None], axis=-1).min(1) if robot.any() else np.full(256, 999.)
    valid = ((paths >= 0) & (paths <= 223)).all((0, 2))
    c = ALGORITHM['candidate']
    positive = valid & (pre <= c['pre_median_motion_max_px']) & (post >= c['post_median_motion_min_px'])
    positive &= (stable >= c['feature_cosine_min']) & (cycle <= c['forward_backward_error_max_px'])
    positive &= (corr >= c['ee_speed_correlation_min']) & (distance <= c['premover_proximity_max_px'])
    background = valid & (pre < .5) & (post < .5)
    score = post/(post+2)*np.exp(-pre/1.5)*np.maximum(stable, 0)*np.maximum(corr, 0)*np.exp(-distance/42)
    return dict(positive=positive, background=background, premover=robot, score=score,
                pre=pre, post=post, stability=stable, cycle_error=cycle, correlation=corr,
                proximity=distance, paths=paths)


def discover(records):
    import numpy as np
    import torch
    import torch.nn.functional as F
    candidates = []
    for r in records:
        if r['audit_only']:
            continue
        for index in np.flatnonzero(r['proxy']['positive']):
            candidates.append((r, int(index), F.normalize(r['features'][3, index], dim=0)))
    selected = []
    while candidates and len(selected) < 4:
        def rank(item):
            r, i, feat = item
            diversity = 1. if not selected else float(1-max(torch.dot(feat, x[2]) for x in selected))
            return (float(r['proxy']['score'][i])*diversity, -r['episode'], -i)
        best = max(candidates, key=rank);selected.append(best)
        candidates = [x for x in candidates if x[0]['episode'] != best[0]['episode']]
    return selected


def bank_bytes(selected):
    import torch
    provenance = []
    for r, i, _ in selected:
        provenance.append(dict(episode=r['episode'], frame=r['frames'][3], patch_index=i,
                               patch_xy=[i % 16, i // 16], pixel_xy_224=[i % 16*14+7, i//16*14+7],
                               source_rgb_sha256=r['rgb_sha256'][3], score=float(r['proxy']['score'][i]),
                               pre_motion=float(r['proxy']['pre'][i]), post_motion=float(r['proxy']['post'][i]),
                               feature_stability=float(r['proxy']['stability'][i]),
                               ee_speed_correlation=float(r['proxy']['correlation'][i]),
                               proximity_to_premover_proxy=float(r['proxy']['proximity'][i]),
                               split='Train35/discovery', semantic_target_verified=False))
    buf = io.BytesIO()
    torch.save({'prototypes': torch.stack([x[2] for x in selected]), 'provenance': provenance}, buf)
    return buf.getvalue(), provenance


def main():
    freeze()
    import av
    import cv2
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    import imageio.v2 as imageio
    torch.manual_seed(1);np.random.seed(1);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    split = json.loads((OUT/'dataset_split_manifest.json').read_text())
    ids = split['train_episode_ids'];assert len(ids) == 35 and not set(ids)&set(split['dev_episode_ids'])
    meta = pd.read_parquet(DATA/'meta/episodes/chunk-000/file-000.parquet')
    tables = []
    for p in sorted((DATA/'data').rglob('*.parquet')):
        df = pd.read_parquet(p);df = df[df.episode_index.isin(ids)]
        if len(df):tables.append(df)
    data = pd.concat(tables)
    hub = Path(torch.hub.get_dir())
    encoder = torch.hub.load(str(hub/'facebookresearch_dinov2_main'), 'dinov2_vits14', source='local', pretrained=False)
    encoder.load_state_dict(torch.load(hub/'checkpoints/dinov2_vits14_pretrain.pth', weights_only=True))
    encoder = encoder.cuda().eval().requires_grad_(False)
    mean = torch.tensor([.485, .456, .406], device='cuda').view(1, 3, 1, 1)
    std = torch.tensor([.229, .224, .225], device='cuda').view(1, 3, 1, 1)
    def encode(images):
        x = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).cuda().float()/255
        with torch.inference_mode():
            return encoder.forward_features((x-mean)/std)['x_norm_patchtokens'].cpu()
    records = [];events = []
    for order, ep in enumerate(ids):
        rows = data[data.episode_index == ep].sort_values('frame_index')
        assert len(rows) and set(rows.task_index) == {24}
        actions = np.stack(rows.action);state = np.stack(rows['observation.state'])
        changes = np.flatnonzero((actions[1:, 6]>.5) & (actions[:-1, 6]<0))+1
        event = int(changes[0]) if len(changes) else None
        if event is None or event < 12 or event+20 >= len(rows):
            events.append(dict(episode=ep, status='NO_COMPLETE_FIRST_CLOSE_WINDOW', event=event));continue
        indices = [event+k for k in ALGORITHM['offsets_frames']]
        row = meta.loc[meta.episode_index == ep].iloc[0];key = 'videos/observation.images.image/'
        path = DATA/'videos/observation.images.image'/f'chunk-{int(row[key+"chunk_index"]):03d}'/f'file-{int(row[key+"file_index"]):03d}.mp4'
        targets = {round(float(row[key+'from_timestamp'])*10)+f:f for f in indices}
        images = {};hashes = {}
        # Decode by frame index in source 10FPS video; avoid approximate seeking across episodes.
        with av.open(str(path)) as stream:
            for n, frame in enumerate(stream.decode(video=0)):
                if n in targets:
                    raw = frame.to_ndarray(format='rgb24')
                    hashes[targets[n]] = hashlib.sha256(raw.tobytes()).hexdigest()
                    images[targets[n]] = cv2.resize(raw[::-1, ::-1], (224, 224), interpolation=cv2.INTER_LINEAR)
                if n >= max(targets):break
        assert len(images) == 9
        images = [images[f] for f in indices]
        features = encode(images);proxy = track(images, features, state[indices])
        r = dict(episode=ep, frames=indices, images=images, features=features,
                 rgb_sha256=[hashes[f] for f in indices], proxy=proxy, audit_only=order % 3 == 2)
        records.append(r)
        events.append(dict(episode=ep, event=event, audit_only=r['audit_only'],
                           positive_patches=int(proxy['positive'].sum()), background_patches=int(proxy['background'].sum()),
                           premover_patches=int(proxy['premover'].sum()),
                           observed_gripper_joint_change=(state[event+12, 6:]-state[event, 6:]).tolist()))
        print('TRAIN_EVENT',ep,event,'CANDIDATES',int(proxy['positive'].sum()),'AUDIT',r['audit_only'],flush=True)
    write('automatic_event_diagnostics.json', events)
    torch.save([{k:v for k,v in r.items() if k != 'images'} for r in records], OUT/'train_only_feature_track_cache.pt')
    selected = discover(records)
    diagnostics = dict(algorithm_sha256=sha(OUT/'prototype_algorithm.yaml'), split_sha256=sha(OUT/'dataset_split_manifest.json'),
                       candidate_count=sum(int(r['proxy']['positive'].sum()) for r in records if not r['audit_only']),
                       candidate_episode_count=sum(bool(r['proxy']['positive'].any()) for r in records if not r['audit_only']),
                       audit_positive_episode_count=sum(bool(r['proxy']['positive'].any()) for r in records if r['audit_only']),
                       selected_count=len(selected), input_episode_ids=ids, dev_image_access=False,
                       semantic_accuracy='UNKNOWN; unlabeled motion/feature proxy only', train_updates=0)
    failures = []
    for key, minimum in [('candidate_count',16), ('candidate_episode_count',4), ('audit_positive_episode_count',4), ('selected_count',4)]:
        if diagnostics[key] < minimum:failures.append(f'{key} below {minimum}')
    if len(selected) == 4:
        first, provenance = bank_bytes(selected)
        # Re-extract ALL Train RGB features and repeat the selection, not just re-save the same tensor.
        repeated = []
        for r in records:
            again = dict(r);again['features'] = encode(r['images'])
            assert torch.equal(again['features'], r['features']), 'DINO extraction not bitwise repeatable'
            repeated.append(again)
        second, provenance2 = bank_bytes(discover(repeated))
        assert provenance == provenance2
        diagnostics['bank_repeat_sha256'] = [hashlib.sha256(first).hexdigest(),hashlib.sha256(second).hexdigest()]
        diagnostics['repeat_bank_identical'] = first == second
        if first != second:failures.append('bank byte repeat mismatch')
        (OUT/'prototype_bank.pt').write_bytes(first)
        write('prototype_provenance.json', provenance)
        bank = torch.stack([x[2] for x in selected]);cos = bank@bank.T
        diagnostics['norms'] = bank.norm(dim=-1).tolist()
        diagnostics['max_off_diagonal_cosine'] = float(cos[~torch.eye(4, dtype=torch.bool)].max())
        if diagnostics['max_off_diagonal_cosine'] > .995:failures.append('prototype feature collapse')
        response = {'positive':[], 'background':[], 'premover':[]}
        for r in records:
            if not r['audit_only']:continue
            scores = (F.normalize(r['features'][3], dim=-1)@bank.T).amax(-1).numpy()
            for kind in response:response[kind].extend(scores[r['proxy'][kind]].tolist())
        for kind in ['background', 'premover']:
            if not response['positive'] or not response[kind]:
                failures.append('missing independent Train audit proxy '+kind);continue
            margin = float(np.quantile(response['positive'],.75)-np.quantile(response[kind],.95))
            diagnostics['audit_positive_q75_minus_'+kind+'_q95'] = margin
            if margin < .02:failures.append('independent Train audit '+kind+' response margin < .02')
        diagnostics['audit_response_distributions'] = {k:dict(count=len(v),quantiles=np.quantile(v,[0,.25,.5,.75,.95,1]).tolist() if v else []) for k,v in response.items()}
        folder = OUT/'prototype_visual_montage';folder.mkdir(exist_ok=True)
        for r, i, _ in selected:
            image = r['images'][3].copy();x=i%16*14;y=i//16*14
            cv2.rectangle(image,(x,y),(x+13,y+13),(255,0,0),1)
            imageio.imwrite(folder/f'ep{r["episode"]}_frame{r["frames"][3]}_auto_patch{i}.png',image)
    diagnostics['failures'] = failures
    diagnostics['status'] = 'G2_AUTO_PROTOTYPE_UNRELIABLE' if failures else 'G2_AUTO_PROTOTYPE_PASS'
    diagnostics['algorithm_must_not_be_retuned_after_gate'] = True
    write('prototype_proxy_diagnostics.json', diagnostics)
    print('AUTO_PROTOTYPE_GATE',diagnostics['status'],failures,flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        (OUT/'automatic_prototype_error.log').write_text(traceback.format_exc())
        raise
