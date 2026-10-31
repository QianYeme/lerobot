"""CPU aggregation, source-locked heatmaps, and existing-rollout evidence only."""
import csv
import json
from pathlib import Path
import sys

ROOT=Path('/root/autodl-tmp/lerobot-main')
R5=ROOT/'experiments/p4_public_benchmark/p4_8_libero_pilot/index5'
OUT=R5.parent/'index6/result'
sys.path.insert(0,str(R5/'workspace/scripts'))
from p48_r5_common import CachedData,sha,write


def read(path):
    with path.open() as stream:return list(csv.DictReader(stream))


def table(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    import cv2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image,ImageDraw
    lock=json.loads((OUT/'input_lock.json').read_text());lookup={s['id']:s for s in lock['samples']}
    data={s:CachedData(s) for s in ['train','dev']}
    folder=OUT/'prototype_heatmaps';folder.mkdir(exist_ok=True)
    for kind in ['legacy','fixed']:
        payload=json.loads((OUT/f'heatmap_payload_{kind}.json').read_text());figures={}
        for entry in payload:
            s=lookup[entry['probe_id']]
            if entry['checkpoint_step']!=10000 or not (s['role']=='bank_reference' or s['episode'] in [813,857]):continue
            sim=np.array(entry['similarity']);weights=entry['region_weights']
            local=np.block([[np.array(weights[1]).reshape(8,8),np.array(weights[2]).reshape(8,8)],
                            [np.array(weights[3]).reshape(8,8),np.array(weights[4]).reshape(8,8)]])
            rgb=np.asarray(data[s['split']].images[s['index'],0]).transpose(1,2,0)
            fig,axes=plt.subplots(1,4,figsize=(12,3.4),layout='constrained')
            axes[0].imshow(rgb,extent=(0,256,256,0));axes[0].set_title('Raw input; reference point only')
            if s['role']=='bank_reference':axes[0].plot(192,120,'r+',markersize=9)
            axes[1].imshow(sim,vmin=-1,vmax=1,cmap='viridis',extent=(0,256,256,0));axes[1].set_title('Actual cosine-max, [-1,1]')
            axes[2].imshow(np.log10(np.maximum(np.array(weights[0]).reshape(16,16),1e-6)),vmin=-6,vmax=0,cmap='magma',extent=(0,256,256,0));axes[2].set_title('Global log10 probability [-6,0]')
            axes[3].imshow(np.log10(np.maximum(local,1e-6)),vmin=-6,vmax=0,cmap='magma',extent=(0,256,256,0));axes[3].set_title('Local conditional log10 prob.')
            for axis in axes:axis.set_xticks([]);axis.set_yticks([])
            fig.suptitle(f'{kind} | {s["split"]} ep{s["episode"]} frame{s["frame"]} | frozen bank; no GT accuracy')
            path=folder/f'{kind}_{s["id"]}.png';fig.savefig(path,dpi=140);plt.close(fig)
            figures[s['id']]=dict(path=str(path.relative_to(OUT)),sha256=sha(path))
        rows=read(OUT/f'prototype_heatmap_index_{kind}.csv')
        for row in rows:
            matching=[x for x in payload if x['probe_id']==row['probe_id']]
            assert len({x['similarity_sha256'] for x in matching})==1,'Frozen DINO map changed between5k/10k'
            row['figure']=figures.get(row['probe_id'],{}).get('path','')
            row['figure_sha256']=figures.get(row['probe_id'],{}).get('sha256','')
            row['source_rgb_sha256']=lookup[row['probe_id']]['rgb_sha256'][0]
            row['figure_scope']='10k image also represents identical5k frozen feature map' if row['figure'] else 'Numerical map retained in heatmap_payload JSON'
        table(OUT/f'prototype_heatmap_index_{kind}.csv',rows)
        # Contact view: raw and similarity for all six preselected stage frames.
        stage=[s for s in lock['samples'] if s['episode'] in [857,813] and s['role']!='bank_reference']
        sheet=Image.new('RGB',(512,len(stage)*280),'white');draw=ImageDraw.Draw(sheet)
        for rank,s in enumerate(stage):
            entry=next(x for x in payload if x['checkpoint_step']==10000 and x['probe_id']==s['id'])
            rgb=np.asarray(data[s['split']].images[s['index'],0]).transpose(1,2,0)
            color=(plt.get_cmap('viridis')((np.array(entry['similarity'])+1)/2)[...,:3]*255).astype('uint8')
            heat=Image.fromarray(color).resize((256,256),Image.Resampling.NEAREST)
            sheet.paste(Image.fromarray(rgb),(0,rank*280+24));sheet.paste(heat,(256,rank*280+24))
            draw.text((4,rank*280+4),f'{kind} {s["split"]} ep{s["episode"]} f{s["frame"]} raw | cosine',fill='black')
        sheet.save(folder/f'{kind}_stage_contact.png')
    metrics={}
    for name in ['object_signal_scales','gate_dose_response','measurement_audit','gradients',
                 'checkpoint_5k_10k_comparison','prototype_heatmap_index','object_dependency_probes']:
        rows=read(OUT/f'{name}_legacy.csv')+read(OUT/f'{name}_fixed.csv')
        for row in rows:
            row.setdefault('probe_id','checkpoint_state');row.setdefault('eval_dtype','float32')
            row.setdefault('units','normalized / raw OSC action');row.setdefault('measurement','r5 reproduction and physical bypass')
        table(OUT/f'{name}.csv',rows);metrics[name]=len(rows)
    summary={}
    for kind in ['legacy','fixed']:
        scales=read(OUT/f'object_signal_scales_{kind}.csv');doses=read(OUT/f'gate_dose_response_{kind}.csv')
        summary[kind]={}
        for step in ['5000','10000']:
            selected=[r for r in scales if r['checkpoint_step']==step and r['split']=='dev']
            values={k:dict(p50=float(np.median([float(r[k]) for r in selected])),p90=float(np.quantile([float(r[k]) for r in selected],.9)),max=max(float(r[k]) for r in selected)) for k in ['query_l2','object_tokens_l2','raw_residual_l2','gated_residual_l2','gated_to_query']}
            values['dose_h0_arm_raw_mean']={label:float(np.mean([float(r['h0_arm_raw']) for r in doses if r['checkpoint_step']==step and r['probe_id'].startswith('dev') and r['dose_label']==label])) for label in ['checkpoint','0.0','0.001','0.01','0.1','-0.01']}
            summary[kind][step]=values
    write(OUT/'quantitative_summary.json',dict(models=summary,csv_rows=metrics,new_training_updates=0,new_rollouts=0))
    # Existing videos/trajectories only: commanded events are not physical grasp labels.
    observations=[];video_dir=OUT/'rollout_evidence';video_dir.mkdir(exist_ok=True)
    rollouts=read(R5/'result/libero_rollouts.csv')
    for kind in ['act','legacy','fixed']:
        sheet=Image.new('RGB',(512*6,280*10),'white');draw=ImageDraw.Draw(sheet)
        selected=[r for r in rollouts if r['kind']==kind]
        for rank,row in enumerate(selected):
            video=Path(row['video']);assert sha(video)==row['video_sha256']
            init=int(row['init_id']);trajectory=json.loads((R5/'result/rollouts'/kind/f'init_{init:02d}_trajectory.json').read_text())
            state=np.array([x['state_before'] for x in trajectory]);action=np.array([x['action'] for x in trajectory])
            closed=np.flatnonzero(action[:,6]>0);close=int(closed[0]) if len(closed) else None
            release=np.flatnonzero((np.arange(len(action))>(close or 0))&(action[:,6]<0)) if close is not None else []
            cap=cv2.VideoCapture(str(video));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));assert count==len(trajectory)
            indices=np.linspace(0,count-1,6).round().astype(int)
            for column,index in enumerate(indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES,int(index));ok,bgr=cap.read();assert ok
                rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);assert rgb.shape==(256,512,3)
                sheet.paste(Image.fromarray(rgb),(column*512,rank*280+24))
                draw.text((column*512+4,rank*280+4),f'{kind} init{init} success={row["success"]} step{index}',fill='black')
            cap.release()
            observations.append(dict(model=kind,checkpoint_step=10000,probe_id=f'init{init}',init_id=init,
                official_success=row['success'],video_path=str(video),video_sha256=row['video_sha256'],video_frames=count,
                command_close_first_step=close,command_open_after_close_first_step=int(release[0]) if len(release) else None,
                eef_z_max_minus_initial=float(state[:,2].max()-state[0,2]),gripper_joint_gap_range=float(np.ptp(state[:,6]-state[:,7])),
                physical_contact_or_grasp='UNKNOWN_NO_OBJECT_EVENT_GROUND_TRUTH',first_failure_stage='UNKNOWN_NOT_INFERRED_FROM_SUCCESS_FLAG',
                units='step index / meters for public EEF and gripper qpos',measurement='existing public-state trace and sampled RGB',eval_dtype='recordedfloat32'))
        sheet.save(video_dir/f'{kind}_all_initial_states_contact.jpg',quality=90)
    table(OUT/'baseline_rollout_observations.csv',observations)
    # Prove the original files are unchanged after forward/backward/counterfactual probes.
    upstream=json.loads((OUT/'upstream_manifest.json').read_text())
    for row in upstream['checkpoints']:assert sha(row['path'])==row['sha256']
    assert sha(upstream['prototype_path'])==upstream['prototype_sha256']
    for name,expected in upstream['source_files'].items():assert sha(name)==expected
    write(OUT/'read_only_verification.json',dict(status='R5_CHECKPOINT_BANK_AND_SOURCE_SHA_UNCHANGED',
         checkpoints_verified=6,new_optimizer_updates=0,new_rollouts=0,model_weights_saved=False))
    print('R6_RENDER_NUMERIC_AND_EXISTING_VIDEO_EVIDENCE_DONE',flush=True)


if __name__=='__main__':main()
