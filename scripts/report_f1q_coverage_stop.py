"""Complete F1Q coverage-stop evidence without fitting on under-covered episodes."""
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

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/force/f1q_load_confound_control/index1/result'


def main():
    gate = json.loads((OUT / 'matching_gate.json').read_text())
    assert not gate['coverage_pass']
    raw = Path(gate['raw'])
    def rows(name):
        with (OUT / name).open(encoding='utf-8-sig', newline='') as stream:
            return list(csv.DictReader(stream))
    coverage, balance = rows('matching_coverage.csv'), rows('balance_audit.csv')
    with (raw / 'matched_pairs.csv').open(encoding='utf-8', newline='') as stream:
        pairs = list(csv.DictReader(stream))
    for r in coverage:
        ps = [p for p in pairs if p['episode'] == r['episode'] and p['role'] == r['role']]
        assert len(ps) == int(r['matched_pairs'])
        assert len({p['pre_frame'] for p in ps}) == len(ps)
        assert len({p['contact_frame'] for p in ps}) == len(ps)
        assert all(float(p['distance']) <= .75 for p in ps)
        with np.load(raw / f'matched_ep{int(r["episode"]):03d}.npz') as z:
            assert len(z['y']) == 2 * len(ps)
            assert np.bincount(z['y'], minlength=2).tolist() == [len(ps), len(ps)]
    groups = {role: sum(int(r['matched_pairs']) for r in coverage if r['role'] == role) for role in ['fit', 'eval']}
    all_eval = [r for r in balance if r['role'] == 'eval' and r['episode'] == 'POOLED_all' and r['stage'] == 'after']
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.axis('off')
    ax.text(.5, .62, 'NOT EVALUABLE: matching coverage gate failed', ha='center', va='center', fontsize=15)
    ax.text(.5, .35, '0/6 fit and 0/6 eval episodes have >=20 pairs.\nNo probes fitted; episode BA deltas and bootstrap CI are NOT_RUN.', ha='center', va='center')
    fig.tight_layout(); fig.savefig(OUT / 'figures/per_episode_delta_NOT_EVALUABLE.png', dpi=150); plt.close(fig)
    metrics = {'scientific_outcome': 'UNRESOLVED_MATCHING_COVERAGE', 'scope': 'EXPLORATORY_ONLY',
               'matched_pairs_total': groups, 'valid_fit_episodes': 0, 'valid_eval_episodes': 0,
               'probe_fits': 0, 'primary_mean_episode_delta': None, 'reason': 'Frozen >=20 pairs per episode and >=4 episodes per group not satisfied',
               'all_eval_matched_balance_descriptive_only': all_eval,
               'F1Q_COMPLETE': False, 'completion_interpretation': 'Section13 explicitly requires probes on valid matched data; not satisfiable at frozen coverage. All authorized matching/audit/stop work delivered, no unauthorized rescue.'}
    (OUT / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'bootstrap.json').write_text(json.dumps({'status': 'NOT_RUN', 'reason': 'No valid matched evaluation episodes', 'replicates_planned': 2000, 'seed': 20261015, 'CI': None}, indent=2))
    (OUT / 'per_episode_metrics.csv').write_text('episode,role,matched_pairs,BA_M0,BA_M1,delta,macro_F1,pre_recall,contact_recall,status\n' +
        ''.join(f"{r['episode']},{r['role']},{r['matched_pairs']},,,,,,,NOT_EVALUABLE_MATCHING_COVERAGE\n" for r in coverage), encoding='utf-8-sig')
    lines = ['# F1Q：匹配覆盖不足，停止于固定门槛', '',
             'SCIENTIFIC_OUTCOME = UNRESOLVED_MATCHING_COVERAGE  \nF1Q_COMPLETE = false  \nEVIDENCE_SCOPE = EXPLORATORY_ONLY  \nNEW_PROBE_FITS = 0  \nNEXT_POLICY_TRAINING_AUTHORIZED = false', '',
             '## 核心结论', '',
             '在冻结F1P的12个episode及原人工接触区间中，严格按4个S0变量、fit-only标准化、同episode、1:1无放回、欧氏距离≤0.75匹配后，**没有任何一集达到20对**。拟合组和验证组均0/6有效，未达到各至少4/6门。不能用小量片段拟合后声称通过，也不能放宽距离救结果。', '',
             '| 组 | Episode | 匹配对数 | 要求 |', '|---|---:|---:|---:|']
    for r in coverage:
        lines.append(f"| {r['role']} | {r['episode']} | {r['matched_pairs']} | ≥20 |")
    lines += ['', f'拟合组共{groups["fit"]}对，验证组共{groups["eval"]}对；不能把跨集加总对数替代逐集覆盖门。无新分类器拟合、无新BA/增量/CI。原F1P的+3.94pp保留，不能将它当作本轮匹配后结果。', '',
              '## 平衡审计', '',
              '逐变量和逐集均值/样本标准差及SMD见balance_audit.csv。将所有小量匹配eval合并的描述性SMD为：', '']
    for r in all_eval:
        lines.append(f"- {r['variable']}: {float(r['SMD']):+.4f}")
    lines += ['', '这些描述性SMD虽均≤0.20，但没有达到覆盖门的eval集合，不能据此越过冻结的判定顺序。主有效集合平衡为不可评估，而不是“残余混杂已排除”。图表展示的是全部小量匹配样本，明确不充当合格集合。', '',
              '## 科学含义', '',
              '当前已有标签中，接触与未接触时的位置/变化/命令条件缺乏足够重叠。F1P探索增量尚不能据此证明在可比运动条件下仍存在；也不能证明增量全由混杂解释或load无用。与匹配覆盖不足相区分，F1P本身的数值没有被改写。', '',
              '## 完成与停止', '',
              '授权范围内匹配、覆盖、平衡、泄漏审计和停止分支已交付。但合同第13节明确要求在有效匹配数据上完成probe；当前不能满足，因此保守标F1Q_COMPLETE=false，不冒充完整probe实验已完成。per_episode BA图标明NOT_EVALUABLE，未伪造数据。', '',
              '下一步仅交用户与规划方验收：若希望继续检验接触专属信息，需新合同决定如何取得位置/运动可比的独立证据；本轮不新增标注/数据、不调整距离、不拟合新模型、不进入F2。', '',
              '无原Dev/Final访问、无policy训练、无架构/部署/真机操作、无推送。原F1/F1P结果保留，D2/D2.1/G1未闭环及B3实际OC-ACT事实不变。']
    (OUT / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (OUT / 'validation.md').write_text('# 验证\n\n- PASS：原子集/标签SHA256对照冻结F1P版本；仅打开其12份W15特征缓存，未加载F1P模型预测或新probe成绩。\n- PASS：配置在匹配前保存；仅4个S0当前通道，fit-only标准化；同集、无放回、距离≤0.75逐对复核。\n- PASS：可靠目标展开沿用F1P，不重新生成/外推标签。\n- NOT_MET：逐集≥20对、每组≥4有效集；实际fit0/eval0。\n- NA：合格eval集合的平衡门；全部小样本的SMD单独描述，不升级结论。\n- NOT_RUN：probe/scaler拟合、增量和bootstrap。匹配变量scaler与probe scaler是两回事。\n- F1Q_COMPLETE=false：第13节probe完成条件无法达成；覆盖停止证据已交付。\n', encoding='utf-8')
    (OUT / 'leakage_audit.md').write_text('# 泄漏审计\n\n仅冻结fit[0,2,4,6,7,9]和eval[10,19,29,38,49,57]。未读原Dev/Final。\n\n匹配距离输入仅F1P S0列[0,4,8,12]（pos、dpos、command[t−1]、dcommand[t−1]）；没有load/current/预测/时间/进度/episode特征。标签只定义可靠候选类；帧号仅确定性遍历/同距解歧，不进入欧氏距离。\n\n匹配标准化只拟合全部可靠pilot-fit W15目标；未用eval统计。候选按episode分别处理，逐对确认两类帧无重复。匹配后probe scaler未拟合，因覆盖不达标。不删除不合格episode，所有12集均列出。\n\n沿用因果模板：obs≤t，命令≤t−1，未跨episode或填补历史；历史采集源码/固件/硬件曝光同步的原局限保留。\n', encoding='utf-8')
    (OUT / 'confound_review.md').write_text('# 混杂控制审查\n\n匹配确实改善已测变量的汇总均衡，但每集只有2–15对，不足以支持冻结评估。不能用整体变量均衡或总样本量替代逐集独立重复。\n\nF1P曾观察到闭爪前与接触后的开口/命令分离；本轮严格匹配的稀少重叠与该观察一致，但不证明因果，也不证明load的正增量完全消失。未做probe，没有匹配后增量结果。\n\n配置包含实现约定：W15通道当前值、标准化在可靠fit目标拟合、CONTACT按源顺序贪心、同距选较早PRE帧。没有尝试其他处理顺序、全局匹配、阈值或变量搜索。\n', encoding='utf-8')
    (OUT / 'proposal.md').write_text('# 待验收：现有数据不能提供足量同情境比较\n\n唯一建议先由用户与规划方决定是否需要、以及如何取得位置/运动条件可比但接触状态不同的证据；选择采集/标注/其它科学控制方式都必须新合同授权。本轮不放宽0.75、不加集/标签、不试其它匹配算法、不加Force head、不训练策略。\n\n同时请验收方澄清第13节：覆盖停止时是否按完整停止分支交付认定完成；当前严格依“probe完成”条款记F1Q_COMPLETE=false。未据此改写原合同。\n', encoding='utf-8')
    shutil.copy2(__file__, raw / Path(__file__).name)
    evidence = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.iterdir() if p.is_file()}
    (raw / 'hash_manifest.json').write_text(json.dumps(evidence, indent=2))
    archive = OUT.parent / 'F1Q_result_20261015.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in OUT.rglob('*'):
            if path.is_file():
                bundle.write(path, path.relative_to(OUT))
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
    print(json.dumps({'outcome': gate['outcome_before_any_probe_performance'], 'total_pairs': groups, 'F1Q_COMPLETE': False, 'zip': str(archive)}))


if __name__ == '__main__':
    main()
