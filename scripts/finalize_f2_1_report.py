"""Assemble the F2.1 r1 result from audited completed records and read-only benchmarks."""
import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from report_f2_1_completed import ROOT, OUT, RAW, main as audit_records


def main():
    audit_records()
    archive=OUT/'training_running_archive'
    if not archive.exists():
        archive.mkdir()
        for name in ['summary.md','validation.md','metrics.json','training_manifest.csv','training_health.md',
                     'implementation_diff.md','mechanism_review.md','latency_and_params.json','proposal.md','leakage_audit.md','baseline_identity.json']:
            shutil.copy2(OUT/name,archive/name)
    audit=json.loads((OUT/'completed_evaluation_audit.json').read_text())
    runtime=json.loads((OUT/'runtime_snapshot.json').read_text())
    benchmark=json.loads((OUT/'selected_benchmark.json').read_text())
    assert benchmark['status']=='PASS'
    assert all(v['complete'] and v['exit_code']=='0' and v['logged_steps']==100000 for v in runtime['variants'].values())
    models=audit['models'];comparisons=audit['comparisons']
    assert comparisons['A1_vs_A0']['percent_change']<0 and comparisons['A2_vs_A1']['percent_change']>0
    assert models['A1']['mechanisms']['zero']['normalized_gripper_change']>0
    outcome='SINGLE_FRAME_LOAD_SUFFICIENT'
    def save(name,value):
        (OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    save('metrics.json',dict(revision='r1',OVERALL_STATUS='PASS',F2_1_COMPLETE=True,SCIENTIFIC_OUTCOME=outcome,
        EVIDENCE_SCOPE='SINGLE_SEED_EXPLORATORY',primary_metric='Dev12 normalized h0 gripper L1, frame weighted; lower better',
        models=models,comparisons=comparisons,Final12_read=False,
        completed_formal_runs=3,failed_previous_A0_attempt=True,diagnostic_model_used=False,
        real_robot_effectiveness='NOT_TESTED',planner_builder_redteam_acceptance='PENDING_EXTERNAL_REVIEW'))
    param0=benchmark['variants']['A0']['parameters']
    cost=dict(benchmark,parameter_increase={name:dict(extra=v['parameters']-param0,
        percent=100*(v['parameters']-param0)/param0) for name,v in benchmark['variants'].items()},
        memory_note='steady-state after warmup; sequential same-process warmup peaks are not comparable; RTX4060 not measured')
    save('latency_and_params.json',cost)
    identity=json.loads((OUT/'baseline_identity.json').read_text())
    identity['r1_selected_checkpoints']={name:dict(checkpoint=m['checkpoint'],step=m['selected_step'],
        model_sha256=benchmark['variants'][name]['model_sha256'],strict_load=True) for name,m in models.items()}
    identity['r1_A0_identity']['checkpoint']=models['A0']['checkpoint']
    save('baseline_identity.json',identity)
    rows=[]
    for name,m in models.items():
        assert benchmark['variants'][name]['checkpoint']==m['checkpoint']
        rows.append(dict(variant=name,status='COMPLETED',optimizer_steps=100000,seed=1,batch_size=4,
            save_freq=20000,selected_step=m['selected_step'],primary=m['primary'],
            training_seconds=m['training_seconds'],attempt=2 if name=='A0' else 1,checkpoint=m['checkpoint']))
    with (OUT/'training_manifest.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    figures=OUT/'figures';figures.mkdir(exist_ok=True)
    def finish(name):
        plt.tight_layout();plt.savefig(figures/name,dpi=160);plt.close()
    labels=list(models)
    for key,name,ylabel in [('primary','a0_a1_a2_primary_metric.png','Normalized h0 gripper L1 (lower better)'),
            ('physical_gripper_mae','gripper_mae_comparison.png','h0 gripper MAE (dataset units)'),
            ('physical_arm_mae','arm_mae_comparison.png','h0 arm MAE (dataset units)')]:
        plt.figure(figsize=(6,4));plt.bar(labels,[models[n][key] for n in labels],color=['#75869b','#258d75','#c98442'])
        plt.ylabel(ylabel);finish(name)
    plt.figure(figsize=(10,4))
    eps=sorted(map(int,models['A0']['per_episode']))
    for name in labels:plt.plot(eps,[models[name]['per_episode'][str(ep)]['primary'] for ep in eps],'o-',label=name)
    plt.xlabel('Dev episode');plt.ylabel('Normalized h0 gripper L1');plt.legend();finish('per_episode_primary_metric.png')
    plt.figure(figsize=(7,4))
    curve=list(csv.DictReader((OUT/'checkpoint_curve.csv').open(encoding='utf-8-sig')))
    for name in labels:
        values=[r for r in curve if r['variant']==name]
        plt.plot([int(r['step']) for r in values],[float(r['primary']) for r in values],'o-',label=name)
    plt.xlabel('Optimizer steps');plt.ylabel('Dev primary metric');plt.legend();finish('checkpoint_selection_curve.png')
    plt.figure(figsize=(8,4))
    for name,relative in [('A0','results_A0_retry1/A0'),('A1','results/A1'),('A2','results/A2')]:
        records=list(csv.DictReader((RAW/relative/'training.csv').open()))
        plt.semilogy([int(r['step']) for r in records],[float(r['loss']) for r in records],alpha=.65,label=name)
    plt.xlabel('Optimizer steps');plt.ylabel('Training total loss (not the primary metric)');plt.legend();finish('training_curves.png')
    for modes,filename in [(['zero'],'load_zero_effect.png'),(['plus_0.1std','minus_0.1std'],'load_perturbation_sensitivity.png')]:
        plt.figure(figsize=(7,4))
        for mode in modes:
            plt.plot(['A1','A2'],[models[name]['mechanisms'][mode]['normalized_gripper_change'] for name in ['A1','A2']],'o-',label=mode+' gripper')
            plt.plot(['A1','A2'],[models[name]['mechanisms'][mode]['normalized_arm_change'] for name in ['A1','A2']],'x--',label=mode+' arm')
        plt.ylabel('Mean absolute normalized action change');plt.legend();finish(filename)
    table='\n'.join(f"| {name} | {m['selected_step']//1000}k | {m['primary']:.6f} | {m['physical_gripper_mae']:.4f} | {m['physical_arm_mae']:.4f} |" for name,m in models.items())
    docs={
    'summary.md':f'''# F2.1 r1 完成：当前证据支持保留单帧load

OVERALL_STATUS=PASS；F2_1_COMPLETE=true；SCIENTIFIC_OUTCOME={outcome}；SINGLE_SEED_EXPLORATORY。

三组均完成100000步，退出码0，各有20/40/60/80/100k五个保存点及完整Dev评估。A0使用用户授权的同seed、同配置重试；原首次非有限梯度失败及15000步未复现诊断完整保留，不将其解释成根因已修复。

## 冻结主指标与选择结果

主指标为固定Dev12所有5748帧的归一化h0夹爪L1，越低越好；无ensemble/相对限幅，不用未来action指导推理。各组从相同五个保存点选主指标最低者，平局较早，不按训练loss挑选。

| 模型 | 选中步数 | 主指标 | 夹爪MAE（原数据单位） | 机械臂MAE（原数据单位） |
| --- | ---: | ---: | ---: | ---: |
{table}

A1相对A0主误差降低15.92%，10/12个Dev episode更好，机械臂误差降低14.88%。A2相对A1主误差增加3.33%，仅5/12 episode更好，机械臂误差增加34.84%。因此本实现/单seed下W15没有稳定额外收益，且有明显臂动作代价；不升级为TEMPORAL_LOAD_PROMISING。A2相对A0夹爪虽改善13.12%，但机械臂反而差14.78%，不能只挑夹爪数字作结论。

## 机制和成本

同观测load置为归一化零基线时，A1/A2夹爪输出平均绝对变化约0.07508/0.06011，说明额外分支确实影响输出，不是完全被忽略。±0.1标准差扰动下，夹爪变化约0.00390–0.00545、机械臂约0.00384–0.00472（均归一化单位）。不把动作变化误称反事实误差改善，也不据此证明真实接触/力估计；整段历史同幅平移不验证时间顺序的使用。

选中checkpoint重新严格加载、batch1有限输出通过；5090纯policy推理中位数约{benchmark['variants']['A0']['cpu_wall_ms_median']:.2f}/{benchmark['variants']['A1']['cpu_wall_ms_median']:.2f}/{benchmark['variants']['A2']['cpu_wall_ms_median']:.2f}ms。10次warmup后50次测量，不含相机/解码/预处理；4060未实测。A0/A1/A2总参数74938375/74955847/74956295，A2比A1仅多448；额外容量仍是讨论因素，但不支持“更多参数就更好”。显存详见latency_and_params.json，区分warmup与稳态，不能用首次cuDNN autotune虚高值比较三组。

## 证据边界与交付

这是同一Dev既选模型又报告结果的探索性离线实验，只有seed1，不能宣称跨seed/新数据稳定，更不能等同真机成功率、抓稳、少洒水或已验证力控。A0首次失败后一次用户授权重试会增加选择/数值稳定性疑虑，额外计算透明列出，原根因未定。历史8D P_hist包含load+current，只作上下文，不充当A0，也不用其策略权重初始化。

推荐保留A1为本轮更简洁候选，先交规划方Builder/Red-Team验收，再决定后续；不自动增加窗口、current、FiLM、contact gate或局部调节器，不进入F2.2。Final12保持封存，无新采/机器人或push。

完整记录、逐episode、机制CSV、15个保存点曲线、8张图、成本和审计在result；源评估/训练日志本地RAW为results/experiments/F2_1_LOAD_TEMPORAL_ENCODING_ABLATION/posttrain_20261019T1418。模型仍在远程，不声称已下载/发布。
''',
    'training_health.md':'''# 训练完成核验

三个有效尝试各100000优化步、batch4、seed1、每20k保存；complete.json和exit=0一致。每组1001条训练日志（step1及每100步），loss/l1/KL/clip前grad norm均有限，最终CSV到100k。run_f2_1每一步检查loss和grad norm非有限即停，没有跳坏batch或关闭检查。

A0首次在14200之后、首20k前非有限梯度退出，精确步与根因未知；有界同配置诊断15000步未复现，用户随后授权一次从0重试。重试未改LR、seed、clip或budget，公共初始hash相同，100k健康退出。失败/诊断从不进入科学模型挑选；不是多个完整成功中择优，但仍增加数值不稳定/条件重试偏差，不能隐藏。A1/A2没有重试。

耗时受并发、GPU占用和起始时间影响：A0重试约2.03h，A1/A2约2.67h，不能以墙钟差直接比较架构速度。最终训练loss A2更低而Dev主指标更差，不应按训练loss选赢家。
''',
    'validation.md':'''# 最终实际验证

准备阶段4项本地pytest及真实GPU batch训练/反向/优化更新、strict reload、6D VAE、current额外键不影响输出、公共初始化hash相同的证据保留。历史P_hist完整5748帧主指标复现与原差1.58e-6<容差1e-4；原配置解析错误及后续通过分别留档。

完成后审核三个有效尝试的exit=0/complete=100k/CSV100k一致；每组五个20k保存点完整。15份评估均5748帧/固定12 Dev episode；逐episode按帧加权复算总主指标一致；selected.json确为五点最小、平局较早。3003条训练日志四项数值有限。

每个选中A1/A2的zero/plus/minus机制记录均覆盖5748帧；导出CSV的汇总按样本数而非batch等权，现batch4恰好整除。合法horizon原评估保留并屏蔽padding。selected checkpoint重新strict加载，state6和load_steps=0/1/15核验，normalization.json与Train-only冻结文件逐字节相同；batch1预测1×100×6全部有限。

5090成本实际测量warmup10+50重复；首次版本显存peak包含cuDNN warmup，保留旧JSON后将统计窗口改为warmup后的稳态，不改模型或主指标。未实测4060、摄像头或真机。新脚本语法/diff及原4项单元测试在本轮重新执行；报告不将工程完成等同真实力控有效。
''',
    'mechanism_review.md':'''# 机制分析：分支被使用，但不证明时间收益

load_zero_ablation.csv及load_perturbation.csv来自选中checkpoint完全相同Dev观测，覆盖5748帧。A1/A2置零的夹爪变化为0.07508/0.06011；臂变化为0.02246/0.01729。zero指normalized load=0（Train均值），不是物理零负载，也不是移除token。

±0.1标准差扰动的夹爪变化约0.00390–0.00545，臂约0.00384–0.00472，未看见归一化均值层面臂响应数量级放大，但不能保证所有帧/关节安全。没有采集真实外力扰动或力标定。

A2仅5/12 episode主指标优于A1，选中点机械臂MAE又差34.84%，不支持时间编码优越。反事实证明计算图的load分支有影响，不证明使用了历史顺序、真正接触、握力或负载因果；load可能是运动/任务阶段代理，F1RM残余混杂仍未排除。
''',
    'builder_review.md':'''# Builder Review证据清单（外部规划方验收仍待进行）

1–5：state_semantics/真实GPU smoke记录共用6D、current排除、A0无load、A1单帧、A2因果W15首帧填充；VAE也6D。6：相同state后单反馈token、同32宽/GELU/512输出，仅1/15输入不同。7：100k/batch4/同LR与clip/seed1，同五点主指标选择；A0失败后用户授权重试单独披露。8–9：固定Train36/Dev12，Final仅元数据。10：原G0公式、缓存归约和历史完整forward复现均保留。11：selected/模型SHA256/严格加载与normalization一致性已核验。12：5090batch1约9.6ms，显存/参数记录完备，4060实际成本未知。

这是执行器自检和供审查的证据，不冒充规划方已经独立批准。没有增加模型、改变阈值或发布checkpoint。
''',
    'red_team_inputs.md':'''# Red-Team待验收的主要风险

- A2仅5/12 episode胜A1，臂误差多34.84%；不能隐藏跨通道代价或只看相对A0的夹爪改善。
- 单seed且Dev用于选checkpoint又报告，存在选择偏差，不是独立Final确认。
- A0第一次非有限梯度后重试，种子相同也不代表轨迹完全确定；原根因未查明。不要把成功重试写成已修复，需讨论稳定性与条件重试偏差。
- 新三组比历史8D P_hist不是严格相同输入/统计/初始化情境，不能把P_hist当A0或由本轮证明current无用。
- load可作任务阶段/运动线索；归一化零/整体历史平移不能证明真实接触或历史顺序使用，F1RM混杂结论不改。
- A2多448参数仍是容量混杂讨论，但本轮不额外加第四个参数匹配模型救结论。
- 5090约9.6ms不等同4060或端到端30FPS，稳态显存不含所有启动峰值/真实相机开销。
- 反事实动作变化不是相对于GT的收益，更不是真机安全/稳定抓取证据。

可支持本轮选择SINGLE_FRAME_LOAD_SUFFICIENT作为探索性工程取舍；不能宣称时间编码普遍无效或已完成力控。规划方正式Red-Team Review尚待进行。
''',
    'proposal.md':'''# 下一步建议（不自动执行）

本轮优先保留A1单帧load，冻结A1 60k选中checkpoint与其normalization/输入语义；A0 80k作干净基线，A2 40k保留反例与审查证据。不要改用各组100k或训练loss最低来改变结果。

先交规划方Builder/Red-Team验收，讨论A0原数值故障、单seed与Dev选择偏差、A2臂代价。若需强化结论，提出新合同决定匹配复验/其他编码，而不是在本轮加current、换W或门控。若后续要真机，须另立部署输入历史与安全门；旧control_act_det不能直接用于这些6D+load新模型。

不自动进入F2.2，不自动做机器人测试、Force head、局部控制、上传HF或push GitHub。Authorization=PENDING_NEXT_DECISION。
''',
    'leakage_audit.md':'''# 最终数据/选择边界审计

仅固定Train36训练及归一化，固定Dev12评估/选模；Final12只查split元数据，不读取样本/视频。F1RM Formal5没有混入。load/state/action统计Train-only，state在模型与VAE均真实6D，current/master排除；A0无反馈token，A1当前load，A2因果W15首帧复制填充，不跨episode。

原8D P_hist只用于历史完整流水线复现，不warmstart新三组、不替代A0。各100k、同seed1/初始化公共hash与超参，在同20/40/60/80/100k上按冻结h0夹爪主指标选择；本次selected正确性和逐episode加权已复算。不用训练loss、100k末点或其它horizon后验换主指标。

同一Dev既选模又报告，所以不是独立泛化确认。A0原失败与同配置诊断不作模型选择候选，用户授权一次从0重试，重试与seed/稳定性风险披露；不隐藏额外计算，不宣称故障根因已修复。eval模式使用zero latent，合法未来action仅作监督目标，不进推理latent；padding排除。M1/M2是同观测离线扰动，不作为接触标签或因果力证据。

没有新采、机器人/部署参数修改、外部发布或push；5090成本不冒充4060/端到端表现。F1RM残余混杂未决、历史P1的输入语义和旧报告均不更改。
'''}
    for name,text in docs.items():(OUT/name).write_text(text,encoding='utf-8')
    required=['summary.md','validation.md','r0_blocked_record.md','baseline_identity.json','state_semantics.json','split_manifest.json',
        'training_manifest.csv','training_health.md','metrics.json','per_episode_metrics.csv','mechanism_review.md',
        'load_zero_ablation.csv','load_perturbation.csv','implementation_diff.md','latency_and_params.json','leakage_audit.md','proposal.md']
    assert all((OUT/name).exists() for name in required)
    manifest={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name!='evidence_manifest.json'}
    for directory in ['figures','variant_configs']:
        for p in (OUT/directory).rglob('*'):
            if p.is_file():manifest[str(p.relative_to(OUT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    save('evidence_manifest.json',manifest)
    destination=OUT.parent/'F2_1_r1_result_20261019.zip'
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as z:
        for p in OUT.iterdir():
            if p.is_file():z.write(p,'result/'+p.name)
        for directory in ['variant_configs','figures','r0_preflight']:
            for p in (OUT/directory).rglob('*'):
                if p.is_file():z.write(p,'result/'+str(p.relative_to(OUT)))
        for p in RAW.rglob('*'):
            if p.is_file() and p.suffix in ['.csv','.json']:z.write(p,'raw/'+str(p.relative_to(RAW)))
        for name in ['run_f2_1.py','retry_f2_1_a0.py','benchmark_f2_1.py','report_f2_1_completed.py','finalize_f2_1_report.py']:
            z.write(ROOT/'scripts'/name,'scripts/'+name)
        for name in ['configuration_act.py','modeling_act.py']:
            z.write(ROOT/'src/lerobot/policies/act'/name,'src/lerobot/policies/act/'+name)
        z.write(ROOT/'tests/scripts/test_f2_1_preparation.py','tests/test_f2_1_preparation.py')
        z.write(OUT.parent/'decision_F2_1_LOAD_TEMPORAL_ENCODING_ABLATION_r1.md','decision_r1.md')
    with zipfile.ZipFile(destination) as z:assert z.testzip() is None
    print(json.dumps(dict(outcome=outcome,complete=True,package=str(destination),bytes=destination.stat().st_size)))


if __name__=='__main__':main()
