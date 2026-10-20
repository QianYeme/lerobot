"""Deliver the manual Formal5 pre-experiment honestly, without claiming frozen F1R completion."""
import csv
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/force/f1r_actuator_load_contact_intervention/index1/result'


def main():
    audit = json.loads((OUT / 'collection_preflight.json').read_text(encoding='utf-8'))
    mapping = json.loads((OUT / 'condition_mapping.json').read_text(encoding='utf-8'))
    assert mapping['mapping'] == {'formal5_kind1': 'FREE_CLOSE', 'formal5_kind2': 'CUP_CONTACT'}
    assert len(audit['trials']) == 10 and sum(r['frames'] for r in audit['trials']) == 1500
    run = ROOT / 'results/experiments/F1R_ACTUATOR_LOAD_CONTACT_INTERVENTION' / ('manual_result_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    run.mkdir(parents=True)
    for name in ['summary.md', 'protocol.json', 'condition_mapping.json', 'collection_preflight.json']:
        shutil.copy2(OUT / name, run / ('previous_' + name))
    def save(name, value):
        (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    def table(name, rows):
        with (OUT / name).open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    operator = {'source': 'USER_EXPLICIT_CONFIRMATION', 'condition_mapping': mapping['mapping'],
                'gripper_control': 'MANUAL_LEADER_TELEOP', 'statement': '是主臂手动控制',
                'formal_frozen_gripper_program': 'NOT_EXECUTED_AS_REQUIRED',
                'data_disposition': 'PRESERVE_AS_MANUAL_PRE_EXPERIMENT_NOT_FORMAL_CONTROLLED_TRIALS',
                'recorded_at_utc': datetime.now(timezone.utc).isoformat()}
    save('operator_confirmation.json', operator)
    protocol = json.loads((OUT / 'protocol.json').read_text(encoding='utf-8'))
    protocol.update({'status': 'DRAFT_NOT_FROZEN_NOT_EXECUTED_AS_FORMAL_PROTOCOL',
                     'observed_collection': operator,
                     'retrospective_freeze_performed': False,
                     'recorded_commands_are_not_the_frozen_program': True})
    save('protocol.json', protocol)
    mapping['command_program_equivalence'] = 'NOT_ESTABLISHED_MANUAL_CLOSURES_NOT_FROZEN'
    save('condition_mapping.json', mapping)
    audit.update({'operator_control_method': 'MANUAL_LEADER_TELEOP',
                  'formal_protocol_status': 'NONCOMPLIANT_MANUAL_CLOSURE_PROGRAM_NOT_FROZEN',
                  'load_current_effect_metrics_computed': False})
    save('collection_preflight.json', audit)
    validity, contacts, per_trial = [], [], []
    for t in audit['trials']:
        trial = f"{t['dataset']}_ep{t['episode']:03d}"
        validity.append({'trial_id': trial, 'condition': t['condition'],
                         'data_integrity': 'FINITE_CONTIGUOUS_REQUIRED_FIELDS_PRESENT',
                         'formal_F1R_validity': 'NOT_VALID_FOR_FROZEN_FORMAL_PROTOCOL',
                         'reason': 'manual leader closure; no identical frozen gripper program',
                         'hardware_safety_or_emergency_stop': 'NOT_INDEPENDENTLY_VERIFIED',
                         'arm_stationarity': 'POSITION_PTP_REPORTED_NO_PREDECLARED_TOLERANCE_OR_FULL_VIDEO_REVIEW',
                         'raw_retained': True, 'discarded': False})
        contacts.append({'trial_id': trial, 'condition': t['condition'], 'first_reliable_contact_frame': '',
                         'last_reliable_contact_frame': '', 'uncertain_start': '', 'uncertain_end': '',
                         'source': 'OPERATOR_CONDITION_CONFIRMED_ONLY',
                         'status': 'CONTACT_INTERVAL_NOT_REVIEWED' if t['condition'] == 'CUP_CONTACT' else 'NO_OBJECT_REPORTED_FULL_VIDEO_NOT_REVIEWED',
                         'preview': f"evidence_samples/{t['dataset']}_ep000_rgb_preview.jpg" if t['episode'] == 0 else '',
                         'reason': 'No contact intervals inferred from load/current; formal program gate failed'})
        per_trial.append({k: v for k, v in {
            'trial_id': trial, 'condition': t['condition'], 'frames': t['frames'],
            'nominal_fps': t['nominal_fps'], 'command_first': t['command_first'], 'command_last': t['command_last'],
            'command_changes': t['command_changes'], 'arm_position_PTP': t['arm_observed_position_peak_to_peak'],
            'load_pre_mean': '', 'load_post_mean': '', 'load_delta': '', 'load_position_matched_effect': '',
            'status': 'EFFECT_NOT_EVALUATED_PROTOCOL_NOT_EQUIVALENT'}.items()})
    table('trial_validity.csv', validity)
    table('contact_annotations.csv', contacts)
    table('per_trial_metrics.csv', per_trial)
    result = {'OVERALL_STATUS': 'PARTIAL_PASS', 'F1R_COMPLETE': False, 'SCIENTIFIC_OUTCOME': 'UNRESOLVED',
              'EVIDENCE_SCOPE': 'EXPLORATORY_SENSOR_CHARACTERIZATION',
              'reason': 'Manual leader control, not the contract-required identical frozen closing program',
              'received_trials': {'FREE_CLOSE': 5, 'CUP_CONTACT': 5},
              'formal_protocol_valid_trials': {'FREE_CLOSE': 0, 'CUP_CONTACT': 0},
              'raw_frames': 1500, 'raw_state_dimensions': 9, 'action_dimensions': 6,
              'load_contact_response': None, 'current_contact_response': None,
              'interprets_zero_valid_as_zero_recorded_or_hardware_failure': False,
              'new_collection_or_robot_operation_performed': False, 'next_policy_training_authorized': False,
              'raw_result': str(run)}
    save('metrics.json', result)
    (OUT / 'validation.md').write_text('# F1R当前交付验收\n\n- 数据接收PASS：固定HF revision，各5集×150帧，必需pos/load/curr、action和双RGB文件存在，原始数值有限、帧号连续。文件哈希见既有audit raw/data_sha256.json。\n- 条件映射PASS（用户明确）：kind1空夹、kind2带杯；两组ep0各4帧预览与此一致，不冒充全视频审核。\n- 固定程序门NOT_MET：用户确认主臂手动闭爪，实际起始/速度/终点也不同，没有合同要求的同一冻结程序。\n- 臂静止/急停/杯损伤/同步：只有现有遥测与局部预览，正式逐集验证尚未完成，不能默认为PASS。\n- 接触区间、load/current效应、位置/命令匹配、接触对齐：NOT_EVALUATED，先停于协议差异；不是load无响应。\n- F1R_COMPLETE=false、科学UNRESOLVED；原数据保留为手动预实验，未删除、重命名为受控正式试验或事后冻结。\n- 无训练、Dev/Final、机器人操作、重采、发布或push。本次只生成result。\n', encoding='utf-8')
    raw_dir = OUT / 'raw'; raw_dir.mkdir(exist_ok=True)
    save('raw/locations.json', {'original_datasets': ['数据集/formal5/formal5_kind1', '数据集/formal5/formal5_kind2'],
         'audit_raw': audit['raw'], 'current_delivery_raw': str(run),
         'raw_signals_not_discarded': True, 'archive_omits_full_videos_and_duplicate_raw_arrays': True})
    (raw_dir / 'README.md').write_text('# 原始数据说明\n\n原HF snapshot、本地数据集和canonical raw均保留，未合并/清洗/删列；locations.json给出位置。紧凑result不重复打包完整视频及原始数组。\n', encoding='utf-8')
    (OUT / 'figures/README.md').write_text('# 图表范围\n\nrecorded_command_equivalence.png为实际十条记录目标曲线；双RGB证据预览在evidence_samples。合同要求的load/current时序、位置对齐、接触对齐图本次NOT_EVALUATED，因为相同冻结程序未执行，且无独立接触区间；未制造空数值图或把手动对比当受控效应。\n', encoding='utf-8')
    (OUT / 'summary.md').write_text('''# F1R result：Formal5手动预实验接收，受控程序门未通过

OVERALL_STATUS = PARTIAL_PASS

F1R_COMPLETE = false

SCIENTIFIC_OUTCOME = UNRESOLVED

EVIDENCE_SCOPE = EXPLORATORY_SENSOR_CHARACTERIZATION

## 一句话结论

**数据已完整接收，但用户确认由主臂手动闭合夹爪，不是F1R要求的同一套冻结闭合程序。因此这10段保留为手动预实验，不能用来宣布受控接触响应实验完成。** 这不是load无用、没有响应或采集文件损坏的结论。

## 实际收到什么

| 仓库 | 用户确认条件 | 集数 | 每集帧数 | 声明FPS |
|---|---|---:|---:|---:|
| QYyyyyyyy/formal5_kind1 | FREE_CLOSE / 空夹 | 5 | 150 | 30 |
| QYyyyyyyy/formal5_kind2 | CUP_CONTACT / 带杯 | 5 | 150 | 30 |

固定revision分别为a6921bdddb862d6937d3104a9274edbe25dcecb7、78ee3801922e7da157247219c0821f32e2164d60。共1500帧，原生9D状态（含master_gripper.pos）、6D action、pos/load/curr及两视角视频保留，没有删列或改写数据。

## 已发现的问题及证据

1. 用户明确确认“是主臂手动控制”，见operator_confirmation.json。
2. 实际记录目标的闭合开始时刻、形状和终点不同，见figures/recorded_command_equivalence.png。空夹最终目标约11.79–18.33，带杯约16.12–23.27；单位是原生归一化位置，不是毫米。单纯轨迹逐点不等不充分证明程序有别，但与人工控制说明结合，固定程序门确实未建立。
3. 各次关节观测多数稳定，最大单关节peak-to-peak约1.4945原生单位。无事前静止容差及完整视频/安全记录，不能据此自动判全部有效或全部硬件异常。
4. 两组ep0第0/60/100/149帧双视角预览已查看，物体有/无与映射一致；尚未审核全部试验或给杯接触起止帧，未用load/current反推标签。

## 当前能说与不能说

这些记录可保留作传感器/采集链路预实验。当前没有报告受控load/current差异、位置或命令匹配效应、4/5同方向响应，不能归CONTACT_RESPONSE_PROMISING，也不能归NO_CLEAR_CONTACT_RESPONSE。不能通过事后时间拉伸、替换闭合协议或按信号挑帧来补造受控结果。

trial_validity.csv逐集说明“对冻结正式协议不合格”，不表示原文件无效、硬件失败或应该删除。正式有效计数为0/0，但收到的试验是5/5，两者不得混淆。所有raw保留；缺失效应指标和接触边界明确NOT_EVALUATED，而不是0。

## 建议的下一步（未执行）

先由用户与规划方验收这份result。若继续完成F1R，需现场安全确定并事先冻结只动夹爪的同一闭合程序，再决定正式采集；已有手动数据不删、不冒充正式试验。没有在本次编写动作脚本、启动采集、重采、发送指令或训练策略。

原F1/F1P/F1Q结果不变；D2/D2.1/G1未闭环和B3实际OC-ACT的更正保持。下一步需要讨论及安全设置，不能直接进入Force策略训练。
''', encoding='utf-8')
    (OUT / 'proposal.md').write_text('# 建议：先验收手动预实验，再确定受控采集安排\n\n若继续F1R，先确认现场安全开合范围并事先冻结同一套夹爪命令程序及日志/视频时序；正式空夹/带杯各5次。保留这批手动记录作为独立预实验，不与后续正式试验混作重复，不依据当前load曲线选择有利闭合范围。\n\n本次只交付result，没有新增采集脚本或执行重采。下一步安排与安全参数待用户确认；不自动开始policy/Force模块/闭环收紧。\n', encoding='utf-8')
    shutil.copy2(__file__, run / Path(__file__).name)
    files = [p for p in OUT.rglob('*') if p.is_file() and p.name != 'evidence_manifest.json']
    save('evidence_manifest.json', {'files': [{'path': p.relative_to(OUT).as_posix(), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in files],
         'full_raw_retained_elsewhere': True, 'result_scope': 'MANUAL_PRE_EXPERIMENT_PROTOCOL_AUDIT'})
    archive = OUT.parent / 'F1R_result_20261016.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for p in OUT.rglob('*'):
            if p.is_file():
                bundle.write(p, p.relative_to(OUT))
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
    print(json.dumps({'result': str(OUT), 'archive': str(archive), 'bytes': archive.stat().st_size,
                      'F1R_COMPLETE': False, 'outcome': 'UNRESOLVED'}))


if __name__ == '__main__':
    main()
