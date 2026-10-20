"""Verify saved F1RM pairs independently and assemble the compact result handoff."""
import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiments/force/f1rm_teleop_load_retrospective_match'
OUT = BASE / 'index1/result'


def read(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def main():
    metrics, freeze, scaler = read('metrics.json'), read('label_freeze.json'), read('matching_scaler.json')
    annotation = OUT / 'contact_annotations.csv'
    digest = hashlib.sha256(annotation.read_bytes()).hexdigest()
    assert digest == freeze['sha256'] == hashlib.sha256((BASE/'contact_annotations.csv').read_bytes()).hexdigest()
    labels = {}
    for r in csv.DictReader(annotation.open(encoding='utf-8-sig')):
        for f in range(int(r['first_reliable_frame']),int(r['last_reliable_frame'])+1):
            assert (r['trial_id'],f) not in labels
            labels[(r['trial_id'],f)] = r['label']
    values = {}
    for kind in ['formal5_kind1','formal5_kind2']:
        d = pd.read_parquet(ROOT/'数据集/formal5'/kind/'data/chunk-000/file-000.parquet')
        for ep in range(5):
            rows = d[d.episode_index==ep].sort_values('frame_index')
            values[f'{kind}_ep{ep:03d}'] = (np.stack(rows['observation.state']),np.stack(rows.action))
    def vector(trial, f):
        s,a = values[trial]
        return np.array([s[f,5],s[f,5]-s[f-1,5],a[f-1,5],a[f-1,5]-a[f-2,5]])
    fit_keys = [(t,f) for (t,f),label in labels.items() if f>=2 and label in ['PRECONTACT','FREE_CLOSE_CLOSING']]
    assert set(scaler['fit_frame_ids']) == {f'{t}:{f}' for t,f in fit_keys}
    fit = np.array([vector(t,f) for t,f in fit_keys])
    mean,scale = np.array(scaler['mean']),np.array(scaler['scale'])
    np.testing.assert_allclose(mean,fit.mean(axis=0),rtol=1e-6)
    np.testing.assert_allclose(scale,fit.std(axis=0),rtol=1e-6)
    pairs = list(csv.DictReader((OUT/'matching_pairs_primary.csv').open(encoding='utf-8-sig')))
    seen_free,seen_cup = set(),set()
    free_vectors,cup_vectors,deltas = [],[],[]
    for r in pairs:
        ft,ff,ct,cf = r['free_trial'],int(r['free_frame']),r['cup_trial'],int(r['cup_frame'])
        assert labels[(ft,ff)]=='FREE_CLOSE_CLOSING' and labels[(ct,cf)]=='CONTACT'
        assert (ft,ff) not in seen_free and (ct,cf) not in seen_cup
        seen_free.add((ft,ff));seen_cup.add((ct,cf))
        distance = np.linalg.norm((vector(ft,ff)-vector(ct,cf))/scale)
        assert distance<=.75 and np.isclose(distance,float(r['distance']),atol=1e-6)
        delta = float(values[ct][0][cf,6]-values[ft][0][ff,6])
        assert delta == float(r['delta_load'])
        assert float(values[ct][0][cf,7]-values[ft][0][ff,7]) == float(r['delta_current'])
        deltas.append(delta);free_vectors.append(vector(ft,ff));cup_vectors.append(vector(ct,cf))
    assert len(pairs)==metrics['passes']['primary']['pairs']==136
    assert np.isclose(np.mean(deltas),metrics['passes']['primary']['load']['mean'])
    a,b = np.array(free_vectors,dtype=float),np.array(cup_vectors,dtype=float)
    smd = (b.mean(axis=0)-a.mean(axis=0))/np.sqrt((a.var(axis=0,ddof=1)+b.var(axis=0,ddof=1))/2)
    audit = list(csv.DictReader((OUT/'balance_audit.csv').open(encoding='utf-8-sig')))
    np.testing.assert_allclose(smd,[float(r['after_smd']) for r in audit],rtol=1e-5)
    validation = dict(status='PASS',label_sha256=digest,paired_frames_verified=len(pairs),
        unique_free_frames=len(seen_free),unique_contact_frames=len(seen_cup),
        checks=['source CSV byte-identical','all scaler frames noncontact','scaler means/SD recomputed',
                'all pairs label-valid','no replacement','all distances recomputed and <=0.75',
                'load/current differences recomputed','pooled mean recomputed','all four SMD recomputed'])
    (OUT/'analysis_validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    docs = {
    'summary.md': '''# F1RM 完成结果：残余混杂未排除

OVERALL_STATUS=PASS；F1RM_COMPLETE=true；SCIENTIFIC_OUTCOME=UNRESOLVED_RESIDUAL_CONFOUND；EVIDENCE_SCOPE=EXPLORATORY_ONLY。

工程完成不代表科研假设通过。本轮只分析现有主臂手动控制的 Formal5 空夹5段+带杯5段，没有训练或新采集。

## 核心结论

**负载存在正向描述性差异，但夹爪位置、运动与目标命令仍不平衡，不能确定差异来自杯子接触而非动作差异。** 不应现在直接给policy加load、Force head或自动收紧控制。

## 标签与样本

用户提交15条RGB审核区间，审核人kg；原CSV字节保留并冻结SHA256。带杯ep001的91–149帧是UNKNOWN，保留且不参与CONTACT；不把它改成接触来凑样本。其余4段可靠接触共283帧。空夹闭合池321帧，带杯PRECONTACT排除前两帧后398帧；scaler拟合719帧，完全不含CONTACT。

所有依据栏为“1”，只有来源而缺少具体视觉理由；这是标签证据的限制，不声称已独立逐帧复审。原始数据/视频哈希与准备阶段一致。

## 主匹配（阈值0.75）

全局跨空夹试次、固定顺序、四变量欧氏距离、1:1无放回。共136对 / 283个接触帧（48.1%），4/5带杯试次各≥10对，覆盖门通过。主覆盖已通过，依冻结配置不执行1.00敏感性；对应CSV仅表头，状态明确NOT_RUN_PRIMARY_COVERAGE_PASSED，不是失败后漏跑。

| 带杯试次 | 匹配对数 | load差均值 | 正差比例 |
| --- | ---: | ---: | ---: |
| ep000 | 61 | +14.89 | 63.9% |
| ep001 | 0 | 不可估计（UNKNOWN） | 不可估计 |
| ep002 | 35 | +18.06 | 85.7% |
| ep003 | 19 | +14.11 | 78.9% |
| ep004 | 21 | +8.57 | 52.4% |

## 平衡门失败

| 四个匹配变量 | 匹配前SMD | 匹配后SMD | 要求绝对值≤0.20 |
| --- | ---: | ---: | --- |
| 夹爪位置 | −1.239 | +0.461 | 失败 |
| 位置变化 | +1.591 | +0.228 | 失败 |
| 上一帧目标命令 | −1.202 | +0.470 | 失败 |
| 上一帧命令变化 | +1.518 | +0.097 | 通过 |

距离≤0.75不保证组间均值平衡。匹配改善了差异，但3/4变量仍越界；这是按合同必须停止升级解释的原因。

## 仅描述性残差与不确定性

load差（杯接触−空夹）均值+14.62、中位数+20.00、样本SD21.26、IQR32.00、正差比例69.9%，全部为原始load单位，不是牛顿。

4个有匹配的带杯试次均值均为正；ep000占44.9%对数、45.7%有符号残差总量。删除任何一个有效带杯试次，均值仍为+13.43至+15.72，因此正方向不依赖单个试次；不等于已排除来源试次及操作者混杂。

按带杯试次cluster bootstrap，2000次、seed20261016，load均值探索性95%CI=[+11.20,+16.89]。只有4个簇，且条件于固定空夹池/固定配对；未包含双侧试次或重匹配不确定性。**CI为正不能挽救平衡门失败，不能用136帧当136个独立试验。**

current仅次要描述：同一136对均值+1.353，cluster CI=[+1.188,+1.589]，不是独立确认，不能替代主load结论。

## 接触起点诊断

只使用原RGB标注首个CONTACT边界，固定前15帧和后15帧（各0.5秒）比较可靠标签内load均值。ep000/002/003/004的后减前分别为−30.13、+35.73、+26.67、+28.27；3/4同向、ep000反向。ep001没有可靠起点。结果并非完全一致，不能称“接触就稳定升高”，也不回改标签或搜索别的窗口。

## 交付及下一步

配对CSV、覆盖/平衡/逐试次表、scaler、bootstrap、起点诊断、8张图及核验报告均在本目录；raw位置见metrics.json。result不附重复视频；视频保留在rgb_review。

交规划方完成Builder Review与Red-Team Review。建议下一轮优先讨论小规模同姿态、同关闭程序、重复空夹/带杯的受控确认；只有新合同及安全确认后才能采集。不要在本轮改标签、阈值、特征或部署，不自动进入调节器/policy训练。本结果也不能证明load无用。
''',
    'confound_review.md': '''# 混杂与试次复现审计

主覆盖通过但3/4匹配变量SMD仍>0.20，最终分类UNRESOLVED_RESIDUAL_CONFOUND。历史命令和位置SMD约0.47/0.46，存在机械/运动解释。所有采集来自手动主臂，非相同冻结程序，不能因匹配后差异为正即推导接触因果。

ep000贡献61/136对和908/1988的有符号差值总和，是最大贡献试次；删除任一有效试次不翻转正方向。没有预注册“占比多大算主导”的阈值，不追加有利阈值。ep000的局部接触前后负载反向，其他三段正向，支持诊断不完全一致。

试次按编号顺序贪心，早试次优先消耗全局空夹池；这是预先固定顺序，不按load重新排序。不可将未匹配CONTACT帧或UNKNOWN的ep001视为随机缺失。独立标签reason均为1，真实边界精度无法从文字复查，不声称完成额外视觉验真。

cluster bootstrap仅按4个CUP试次，条件于固定FREE池，不处理共享FREE来源的双向簇依赖。CI为正不排除混杂。没有真实力标定、因果效应或真机控制收益证据。
''',
    'leakage_audit.md': '''# 泄漏审计

用户通过仅显示RGB的页面产生15条标注，在任何本轮load/current效应之前提交。先验证区间/标签/试次，保存原CSV及SHA256、固定匹配配置和脚本快照；未修改用户UNKNOWN或边界。来源声明为USER_SUBMITTED_RGB_REVIEW，不声称代码可证明人的盲法遵守或边界准确。

仅读取Formal5 kind1/kind2内容和父F1R清单。未读取原Formal4 Dev12/Final12样本内容。四变量由pos[t]、pos[t]-pos[t-1]、action[t-1]、action[t-1]-action[t-2]组成，不含force/current、绝对进度、ID特征或未来命令。ID只作确定性遍历。

标准化拟合719帧，仅FREE闭合和PRECONTACT；CONTACT不参与。UNKNOWN、未标注、前两帧排除。每一pass一次固定greedy，不放回；匹配结束后才读取被配对load/current并计算残差。主覆盖通过因此未执行1.00，不用敏感性替代主结论。

接触起点诊断固定15帧前后，不调标签、不搜索窗口；current在完全相同配对上次要报告。bootstrap按试次重采样而非帧。两阶段源码/标签哈希和验证文件可追溯；未训练、部署、重采、操作机器人或推送。
''',
    'validation.md': '''# 完成验证

1. 准备阶段10段1500帧结构/有限性/连续帧与时间验证、两相机3000帧解码、10段30FPS审核视频验证通过；本次重验源文件SHA256一致及state/action名字与顺序。
2. 用户15条区间覆盖10试次：整数端点、范围、无重叠、条件兼容、非空审核人/依据检查通过。原CSV与result快照字节相同，label_freeze.json记录SHA256；UNKNOWN保持原样。reason均为1，语法通过不等于视觉标签已独立验真。
3. 7项定向pytest通过：历史命令无未来泄漏、力列不入特征、greedy阈值/不放回/并列顺序、SMD退化、UNKNOWN/条件与重叠校验、cluster bootstrap确定性。
4. report_f1rm.py独立读取原始parquet复算：scaler拟合集合、均值/标准差、136对标签、唯一性、全部距离≤0.75、load/current差值、池均值和四个SMD一致；analysis_validation.json为PASS。不重新匹配、不另选阈值。
5. 1.00敏感性未运行是冻结条件（仅覆盖不足）未触发；其CSV表头不代表已有结果。136对来源唯一，5个CUP试次均列出；ep001缺可靠CONTACT不填0效应。
6. 8张图生成，平衡图已人工视觉检查；原页面9项浏览器测试保持为准备阶段证据，与科学门分开。

工程完成PASS；覆盖PASS；平衡FAIL；科学UNRESOLVED_RESIDUAL_CONFOUND。F1RM_COMPLETE=true只表示授权分析交付完整，绝不等于有用信号确认。
''',
    'proposal.md': '''# 提议：小型受控确认，而非立即加Force模块

## Observation

136对和4个有效带杯试次提供正向描述性load差，但位置/运动/命令平衡未达标，起点诊断有1/4反向。残差证据尚不能剥离手动操作混杂。

## Proposed Change

由规划方与用户讨论一个新受控合同：固定机械臂姿态、相同安全关闭程序，重复空夹与带杯；顺序交错或随机化、留存独立RGB接触标签与命令记录。先解决可比性，再考虑调节器。安全范围、终点、速度和退出行为须现场确认；本提议没有执行授权。

## Risks / alternatives

透明杯视觉边界、低分辨率、机械摩擦/迟滞、杯形与水量都可能影响比较。不要为得到显著性在本轮重标UNKNOWN、换窗口、搜索特征或调整阈值。若不愿新采，当前只能保留探索性未决结论，不能承诺数据重算必能排除混杂。

## Review handoff

Builder应核验历史命令与原始schema；Red-Team应挑战残余平衡、标签reason=1、4簇CI和空夹来源依赖。两项Review由规划方完成，此处不冒充已验收。未经新decision及用户批准，不开始任何采集、Force head、自动收紧、policy训练或部署。

Authorization = PENDING USER APPROVAL.
'''}
    for name, content in docs.items():
        (OUT/name).write_text(content,encoding='utf-8')
    required = ['summary.md','validation.md','trial_manifest.csv','contact_annotations.csv','matching_config.json',
        'matching_scaler.json','matching_pairs_primary.csv','matching_pairs_sensitivity.csv','matching_coverage.csv',
        'balance_audit.csv','per_trial_metrics.csv','metrics.json','bootstrap.json','confound_review.md','leakage_audit.md','proposal.md']
    assert all((OUT/name).exists() for name in required)
    assert len(list((OUT/'figures').glob('*.png')))==8
    raw = Path(metrics['raw'])
    shutil.copy2(__file__,raw/'report_f1rm.py')
    manifest = {str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in OUT.rglob('*') if p.is_file() and p.name!='evidence_manifest.json'
        and 'rgb_review' not in p.parts}
    (OUT/'evidence_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    for name in required+['analysis_validation.json','evidence_manifest.json']:
        destination = raw/name
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).digest() == hashlib.sha256((OUT/name).read_bytes()).digest():
                continue
            destination = raw/('final_'+name)
        shutil.copy2(OUT/name,destination)
    archive = BASE/'index1/F1RM_result_20261018.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name in required+['analysis_validation.json','evidence_manifest.json','label_freeze.json',
                              'contact_onset_diagnostic.csv','input_validation.json','browser_smoke.json']:
            z.write(OUT/name,'result/'+name)
        for p in sorted((OUT/'figures').glob('*.png')):
            z.write(p,'result/figures/'+p.name)
        z.write(BASE/'index1/decision_F1RM_TELEOP_LOAD_RETROSPECTIVE_MATCH.md','decision_F1RM.md')
        for name in ['analyze_f1rm.py','report_f1rm.py']:
            z.write(ROOT/'scripts'/name,'scripts/'+name)
        z.write(ROOT/'tests/scripts/test_f1rm.py','tests/test_f1rm.py')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    print(json.dumps(dict(validation=validation,archive=str(archive),bytes=archive.stat().st_size)))


if __name__=='__main__':
    main()
