"""Publish actual dense-window review coverage, preserving prior partial reports."""
import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import cv2

from analyze_d2_1_dense import DEV, OUT, RAW


def save(name, value):
    path = OUT / name
    if path.exists():
        history = OUT / 'prior_sampled_report'
        history.mkdir(exist_ok=True)
        previous = history / name
        if not previous.exists():
            shutil.copy2(path, previous)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    manifest = json.loads((RAW / 'dense_review/manifest.json').read_text())
    evidence = json.loads((OUT / 'manual_evidence_manifest.json').read_text())
    assert len(manifest) == len(evidence) == 48
    assert {x['episode'] for x in manifest if x['split'].lower() == 'dev'} == set(DEV)
    final = {1, 8, 11, 16, 20, 27, 31, 37, 44, 46, 52, 59}
    assert not final.intersection(x['episode'] for x in manifest)
    clip_checks = []
    for item in evidence:
        for clip in item['clips']:
            cap = cv2.VideoCapture(clip['path'])
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            size = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
            cap.release()
            assert count == clip['frame_end'] - clip['frame_start'] + 1
            assert abs(fps - 30) < .01 and size == [1280, 480]
            clip_checks.append({'episode': item['episode'], 'frames': count, 'path': clip['path']})
    events, reviews = [], []
    for item in manifest:
        ep = item['episode']
        start = 120 if ep in DEV or ep in (0, 2, 4, 6) else 200
        upper = DEV[ep][0][1] if ep in DEV else 450
        folder = f'manual_review/ep{ep:03d}'
        reason = ('Top view does not expose cup-bottom/table gap; wrist view moves with arm. '
                  'Transparent cup and occluded contact plane prevent reliable physical lift-off/contact/hold boundaries. '
                  'Ordered frame sheets show pose change and return, not calibrated vertical height.')
        reviews.append({**item, 'reviewed_window': [start, item['decoded_frames_per_camera'] - 1],
                        'status': 'DENSE_ORDERED_FRAME_SHEETS_REVIEWED_BY_CODEX',
                        'continuous_video_playback_by_codex': False,
                        'individual_full_resolution_every_frame_review': False,
                        'human_reviewed': False, 'released_by_frame': upper,
                        'physical_lift_hold': 'UNKNOWN', 'reason': reason})
        base = {'episode': ep, 'split': item['split'], 'source': 'DENSE_ORDERED_FRAME_SHEETS_REVIEWED_BY_CODEX',
                'human_reviewed': False, 'clip': f'{folder}/clamp_lift_hold_return.mp4',
                'evidence_images': f'{folder}/frame_250.jpg;{folder}/frame_280.jpg;{folder}/frame_340.jpg'}
        for event in ['effective_clamp', 'lift_off', 'descent_start', 'table_contact', 'top_hold']:
            events.append({**base, 'event': event, 'status': 'UNKNOWN_AFTER_WINDOW_REVIEW',
                           'frame': None, 'time_s': None, 'search_start': start, 'search_end': 359,
                           'reason': reason if event != 'effective_clamp' else
                           'Visible jaw closure/cup coupling is not proof of effective force-bearing clamp; early closure window not densely reviewed for all Train.'})
        events.append({**base, 'event': 'released_state', 'status': 'VISIBLE_RELEASED_BY_FRAME_NOT_ONSET',
                       'frame': upper, 'time_s': upper / 30, 'search_start': 330,
                       'search_end': item['decoded_frames_per_camera'] - 1,
                       'clip': f'{folder}/contact_release.mp4',
                       'evidence_images': f'{folder}/frame_450.jpg',
                       'reason': 'Open jaws separate from cup; upper bound only, not exact release onset or proof of prior lift.'})
    with (OUT / 'demonstration_events.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(events[0]))
        writer.writeheader()
        writer.writerows(events)
    save('visual_review_manifest.json', reviews)
    save('demonstration_audit.json', {'dev_dense_window_reviews': 12, 'train_dense_window_reviews': 36,
         'full_episode_continuous_playback_reviews': 0, 'released_state_observations': 48,
         'reliable_physical_lift_hold_intervals': 0, 'review_method': 'ordered every-frame grids, not 1 fps',
         'limitations': 'Most Train review starts at frame 200; Dev starts at 120. No complete video playback claim.'})
    dense = json.loads((OUT / 'dense_phase_metrics.json').read_text())
    aggregate = dense['frame_weighted_aggregate']
    metrics = {'DIAGNOSIS_COMPLETE': False, 'acceptance_status': 'PARTIAL_PASS', 'diagnostic_category': 'UNRESOLVED',
               'B3_identity_status': 'OCACT_VERIFIED', 'dev_episodes': 12, 'dev_frames': 5748,
               'dev_dense_reviews': 12, 'train_dense_reviews': 36,
               'reliable_released_core_frames': aggregate['raw_h0']['released_retreat_core']['frames'],
               'physical_lift_hold_evaluated': False, 'full_chunk_ground_truth_evaluated': True,
               'frame_weighted_aggregate': aggregate, 'training': False, 'final_test': False, 'real_robot': False}
    save('metrics.json', metrics)
    save('prediction_diagnosis.json', {'category': 'UNRESOLVED', 'teacher_forced': True,
         'physical_hold_evaluated': False, 'available': 'full task and conservative released-retreat core',
         'candidate_posture_window': 'auxiliary only; NOT airborne hold', 'metrics': 'dense_phase_metrics.json'})
    save('postprocess_replay.json', {'configurations': ['off+5', 'off+2', '0.01+5', '0.01+2'],
         'reference': 'raw_h0', 'state': 'recorded expert state, not fresh motor feedback',
         'aggregate': aggregate, 'physical_hold_causality_established': False,
         'equivalence_evidence': 'replay_equivalence_validation.json'})
    save('validation.json', {'acceptance_status': 'PARTIAL_PASS', 'DIAGNOSIS_COMPLETE': False,
         'cache_validation': 'dense_cache_validation.json', 'clip_frame_count_fps_resolution': 'PASS',
         'checked_clips': len(clip_checks), 'fixed_split_no_final': 'PASS',
         'B3_identity': 'OCACT_VERIFIED', 'physical_lift_hold_boundaries': 'UNRESOLVED',
         'continuous_playback_not_performed': True, 'no_training_deployment_change_or_robot': True})
    save('report_consistency_audit.json', {'status': 'PASS_WITH_EXPLICIT_GAPS',
         'diagnosis_complete': False, 'B3_plain_ACT_claim': False, 'hold_duration_claim': False,
         'full_chunk_ground_truth_evaluated': True, 'primary_phase': 'released_retreat_core',
         'release_window_is_not_onset_bracket': True,
         'historical_sampled_reports': 'prior_sampled_report/; superseded, not current coverage'})
    lines = ['# D2.1 密集复核与缓存分析交付', '',
             'DIAGNOSIS_COMPLETE=FALSE；acceptance_status=PARTIAL_PASS；diagnostic_category=UNRESOLVED；B3_identity_status=OCACT_VERIFIED。', '',
             '## 本次实质进展', '',
             '固定 Dev12 第120帧至结尾、Train36 第200帧至结尾已按双视角连续帧序列复核；Train 0/2/4/6亦复核120–199帧。不是每秒一帧抽审。全部48集连续解码、96段原帧号双视角片段及480张全分辨率证据图已生成。', '',
             '当前工具以图片查看：实际审核是逐帧有序联系表与部分全分辨率图，**不是已连续播放完整视频，也不是每一帧全分辨率精审**。因此未满足合同的完整视频审核要求；没有把未执行项目写成已完成。', '',
             '48集均可见释放后的夹爪—杯子分离；事件表记录“截至某帧已释放”，不是释放起点。闭爪与杯子姿态变化可见，但透明杯底、顶视接触面不可见及腕部相机随臂移动，使离桌、触桌和真正顶部保持边界仍不可判。不能把关节静止或姿态平台当作空气中保持。', '',
             f'P1全部12Dev/5748帧原始h0和四配置分析完成：可靠释放后退回区间{metrics["reliable_released_core_frames"]}帧，其余物理阶段单列unknown。300行阶段/辅助窗口数值见per_episode_phase_metrics.csv；辅助候选姿态窗口不参与hold结论。', '',
             '| 配置 | 全任务机械臂MAE | 全任务夹爪MAE | 已释放退回机械臂MAE | 已释放退回夹爪MAE |',
             '|---|---:|---:|---:|---:|']
    for config, value in aggregate.items():
        full, core = value['full_task'], value['released_retreat_core']
        lines.append(f'| {config} | {full["arm_mae"]:.4f} | {full["gripper_mae"]:.4f} | {core["arm_mae"]:.4f} | {core["gripper_mae"]:.4f} |')
    lines += ['', '单位为数据集原生关节单位，不是毫米或牛顿；各关节MAE、动作相对state差值、裁剪率、处理差值均在JSON/CSV。回放以录制专家state为条件，不是自主闭环。后处理会改变命令，但现有证据无法归因真实低抬升或短保持。', '',
              '## 保留的核验与剩余缺口', '',
              'B3配置/参数/module/实际forward身份核验及归一化回放等价结果保留；最大差9.918e-5。缓存维度、有限性、帧连续性和旧D2回放MAE复现已通过。合法horizon 1/5/10/30/100见horizon_metrics.json。原D2未覆盖；先前部分交付zip保留，部分JSON旧版本另存prior_sampled_report。', '',
              '缺口：完整连续视频精审、可靠物理离桌/顶部保持/触桌边界、真实runtime记录。既没有证据判定示范无hold，也没有证据认定后处理为唯一原因。', '',
              '唯一主要下一步：先按manual_review.md复核Dev 3/28/58的现成双视角片段，只回答接触面是否可辨及可辨边界；不要求重做48集，不训练、不改部署、不追加真机。若三段仍不可辨，停止从这些像素强推精确hold标签，交网页GPT验收证据边界。', '',
              '无训练、无Final内容、无真机、无push；G1正式评价complete仍FALSE。']
    (OUT / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (OUT / 'proposal.md').write_text('# 唯一下一步：现成视频可见性最小人工复核\n\n先复核Dev3/28/58；可辨则补物理事件并重算同一P1缓存，不训练。若接触面仍不可见，则反证这些视频足以提供精确hold真值；不得据此判定无hold或部署根因。详见manual_review.md。\n', encoding='utf-8')
    manual = ['# 最小人工复核：无需真机或重采', '',
              '先看以下3集。左侧top、右侧腕部；画面红字是原episode帧号。时间=帧号/30；片段内部0秒分别对应原第120/330帧。', '',
              '问题：能否直接看见杯底与桌面分离/再次接触？能否确认离桌后且下降前有顶部停留？若能给最早/最晚边界帧；若不能直接写“接触面不可见”。不要从水面大小、阴影、关节电流或腕部背景运动推定高度。', '']
    for ep in (3, 28, 58):
        folder = f'manual_review/ep{ep:03d}'
        manual += [f'## Dev episode {ep}', '',
                   f'[闭爪—候选抬升—返回，120–359帧（4.000–11.967秒）]({folder}/clamp_lift_hold_return.mp4)', '',
                   f'[触桌/释放候选，330–478帧（11.000–15.933秒）]({folder}/contact_release.mp4)', '',
                   f'[250帧证据图]({folder}/frame_250.jpg)、[280帧证据图]({folder}/frame_280.jpg)、[340帧证据图]({folder}/frame_340.jpg)。', '',
                   '不可判原因：顶视不能直接看见桌面间隙，夹爪遮挡透明杯底，腕部视角随臂移动；可见姿态变化不足以独立确定物理接触边界。', '']
    manual += ['其余45集同目录已有对应片段；完整窗口、原因和图路径见demonstration_events.csv及manual_evidence_manifest.json。当前没有人工确认，勿标HUMAN_REVIEWED。']
    (OUT / 'manual_review.md').write_text('\n'.join(manual) + '\n', encoding='utf-8')
    files = [p for p in OUT.iterdir() if p.is_file() and p.suffix in ('.json', '.csv', '.md')]
    save('evidence_manifest.json', {'files': [{'path': p.name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in files if p.name != 'evidence_manifest.json'],
         'raw_frames': str(RAW / 'dense_review/manifest.json'), 'manual_clips': clip_checks})
    archive = OUT.parent / 'D2_1_dense_result_20261014.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in OUT.iterdir():
            if path.is_file() and path.suffix in ('.json', '.csv', '.md'):
                bundle.write(path, path.relative_to(OUT))
        for ep in (3, 28, 58):
            for path in (OUT / f'manual_review/ep{ep:03d}').iterdir():
                bundle.write(path, path.relative_to(OUT))
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
    print(json.dumps({'reviewed_episodes': 48, 'validated_clips': len(clip_checks), 'reliable_core_frames': metrics['reliable_released_core_frames'], 'diagnosis_complete': False}))


if __name__ == '__main__':
    main()
