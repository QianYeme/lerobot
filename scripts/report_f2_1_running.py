"""Update local F2.1 handoff from an actual remote status snapshot, never launch jobs."""
import csv
import hashlib
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/force/f2_1_load_temporal_encoding_ablation/index1/result'


def main():
    snapshot=json.loads((OUT/'runtime_snapshot.json').read_text(encoding='utf-8'))
    archive=OUT/'r1_pretraining'
    if not archive.exists():
        archive.mkdir()
        for name in ['summary.md','validation.md','metrics.json','implementation_diff.md','proposal.md','leakage_audit.md','baseline_identity.json']:
            shutil.copy2(OUT/name,archive/name)
    def save(name,value):
        (OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    for key,expected in snapshot['source_sha256'].items():
        if key.startswith('source/'):
            path=ROOT/key.removeprefix('source/')
            content=path.read_bytes()
            assert hashlib.sha256(content).hexdigest()==expected or hashlib.sha256(content.replace(b'\r\n',b'\n')).hexdigest()==expected,key
    parquet=ROOT/'数据集/formal4/kind_merged_nomaster/data/chunk-000/file-000.parquet'
    assert hashlib.sha256(parquet.read_bytes()).hexdigest()==snapshot['dataset_parquet_sha256']
    assert snapshot['smoke']['status']=='PASS' and snapshot['baseline_forward']['status']=='PASS'
    smoke=snapshot['smoke']['variants']; count0=smoke['A0']['parameters']
    failed=[name for name,v in snapshot['variants'].items() if v['exit_code'] not in (None,'0')]
    overall='PARTIAL_TRAINING_FAILURE' if failed else 'RUNNING'
    rows=[]
    for name,variant in snapshot['variants'].items():
        assert variant['last_log'] and float(variant['last_log']['loss'])==float(variant['last_log']['loss'])
        rows.append(dict(variant=name,status='RUNNING' if variant['exit_code'] is None else ('COMPLETED' if variant['exit_code']=='0' else 'FAILED'),screen=variant['screen'],
            observed_step=variant['logged_steps'],target_steps=100000,seed=1,batch_size=4,save_freq=20000,
            initialization_hash=smoke[name]['shared_hash'],snapshot_utc=snapshot['timestamp_utc'],
            output_dir=variant.get('output_dir',snapshot['base']+'/results/'+name)))
    with (OUT/'training_manifest.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save('metrics.json',dict(revision='r1',OVERALL_STATUS=overall,F2_1_COMPLETE=False,SCIENTIFIC_OUTCOME='UNRESOLVED',
        training_runs=3,completed_runs=0,EVIDENCE_SCOPE='SINGLE_SEED_EXPLORATORY',
        snapshot_utc=snapshot['timestamp_utc'],variants=snapshot['variants'],
        formal_Dev_comparison='PENDING_FIXED_CHECKPOINTS',Final12_read=False))
    save('latency_and_params.json',dict(gpu=snapshot['gpu'],parameters={name:dict(
        total=s['parameters'],trainable=s['trainable_parameters'],extra_vs_A0=s['parameters']-count0,
        percent_increase=100*(s['parameters']-count0)/count0,smoke_peak_vram_bytes=s['peak_vram_bytes']) for name,s in smoke.items()},
        A2_extra_vs_A1=smoke['A2']['parameters']-smoke['A1']['parameters'],
        latency='PENDING_SELECTED_CHECKPOINT_BENCHMARK; smoke wall time is not inference latency',
        RTX4060_deployability='NOT_MEASURED; no real-robot claim'))
    identity=json.loads((OUT/'baseline_identity.json').read_text(encoding='utf-8'))
    identity['r1_A0_identity']=dict(config='variant_configs/A0.json',shared_hash=smoke['A0']['shared_hash'],parameters=count0,
        strict_reload='PASS_REAL_BATCH_GPU',initialized_without_P_hist=True,checkpoint='training not finished')
    save('baseline_identity.json',identity)
    docs={
    'summary.md':f'''# F2.1 r1：训练状态快照

OVERALL_STATUS={overall}；F2_1_COMPLETE=false；SCIENTIFIC_OUTCOME=UNRESOLVED；SINGLE_SEED_EXPLORATORY。

截至UTC {snapshot['timestamp_utc']}：{'; '.join(f"{name}: step={v['logged_steps']}, exit={v['exit_code']}" for name,v in snapshot['variants'].items())}。失败组：{failed}。screen实际状态见runtime_snapshot.json，不把已退出的组说成运行中；未自动重启或调整参数。实际GPU：{snapshot['gpu']}。服务器connect.weste.seetacloud.com:53289，运行目录{snapshot['base']}；旧远程dirty工作区未覆盖。

当前尝试记录：{json.dumps({k:dict(attempt=v.get('attempt',1),output=v.get('output_dir'),previous=v.get('previous_attempt')) for k,v in snapshot['variants'].items()},ensure_ascii=False)}。A0若为retry1，是用户明确要求的同seed/同参数从0重试，不覆盖原失败，也不把诊断模型纳入比较；本轮不自动反复重启。

## 已通过的门

- 原始parquet SHA256与本地一致，prototype SHA256=114de07e6cf6c0a2eba7fbede910212e0be6b0740751ab42855780c63b5fb876。只用Train36/Dev12；Final只查split元数据，未读取样本。
- 三组共同6D位置、无current/master；A0不接收load，A1仅当前归一化load，A2因果W15首帧左填充。load/state/action统计仅Train。VAE的状态投影也为6D。
- 真实batch4、chunk100训练forward/backward、有限梯度/裁剪、非零load层梯度、optimizer参数更新、strict checkpoint重载和current额外键不影响输出检查通过。Dev指标、合法horizon与M1/M2入口做实际batch smoke。
- 三组公共初始化SHA256完全一致：{smoke['A0']['shared_hash']}。不使用P_hist策略权重warmstart；共享视觉/object预训练来源相同。
- 完整历史P_hist的5748帧Dev主指标重跑为{snapshot['baseline_forward']['score']:.11f}，原G0为{snapshot['baseline_forward']['reference']:.11f}，差约1.58e-6，低于预设1e-4容差；这次不是仅缓存聚合。

## 正式运行协议

三组均100000 optimizer steps、batch4、seed1、AdamW、lr1e-5/backbone1e-4、weight_decay1e-4、clip10、无scheduler、无AMP。每20000步保存并完整评估Dev12，按归一化h0夹爪L1最低选checkpoint，平局较早。

A1/A2使用相同位置的额外load token，MLP1或15→32→512/GELU，新增512维位置embedding，always-on无gate。总参数A0=74938375，A1=74955847，A2=74956295；A2比A1仅多448。确切计数见latency_and_params.json。延迟与RTX4060实测尚未完成，smoke秒数不能冒充推理延迟。

每次固定checkpoint评估还输出物理单位夹爪/臂误差、逐Dev episode主指标、合法1/5/10/30/100目标horizon，及相同观测上的load置零与±0.1标准差离线扰动（不是物理力扰动）。主指标不因次指标改变。

当前screen和输出路径以training_manifest.csv/active_runs.json为准。A0原尝试日志为A0.log，重试为A0_retry1.log/results_A0_retry1/A0；A1/A2仍为原日志与results目录。结束写对应exit文件，SSH断开不依赖Codex触发。每20k checkpoint保存在各自output_dir/checkpoints/020000/pretrained_model等；latest_optimizer.pt只保留最近一次优化器状态，不宣称实现完整可复现断点恢复。

## 当前边界

现在只有早期训练健康证据，没有完成的模型比较，不可根据训练loss宣布A2更好。所有科学结论等待固定checkpoint评估和机制分析。预计时长应按后续稳定吞吐更新；约0.1秒/组/步的启动观测对应纯训练约2.8小时，另加5轮完整Dev/机制评估，暂估总3–4小时，不是保证。

原r0见r0_preflight；r1数据准备报告见r1_pretraining。没有机器人操作、重采、Final样本、部署参数修改或push，也不进入F2.2。
''',
    'validation.md':'''# r1 已执行验证

本地4项数据/因果窗口/h0指标pytest通过。GPU实际证据见gpu_smoke.json：三组真实batch训练、backward/非零反馈梯度、有限梯度裁剪、optimizer更新、6D VAE、额外current键不影响输出、strict reload，以及Dev指标/horizons/M1/M2 batch入口通过。公共初始化hash完全一致。

历史P_hist完整Dev5748帧重跑通过，score0.059309560033，对比原0.059307981561，容差1e-4。配置加载最初使用ACTConfig.from_pretrained遇type字段解析错误；改为既有PreTrainedConfig入口后成功，失败日志留在远程gates_attempt1.log，没有改变checkpoint。后续补optimizer及评估入口smoke均保留之前快照，不把首次失败写成通过。

远程pytest包未安装，所以没有声称在远程运行pytest通过；模型级验证是独立runner断言，数据单元测试在本地实际运行。旧源码混合行尾在打包时归一化，恢复本地原混合行尾后仅27行新增；实际远程源码哈希和本地归一化行尾哈希核验一致，无无关格式重排。remote parquet和prototype哈希一致。

三组screen启动后各有实际训练CSV记录、有限loss和grad_norm，无退出码。不能把screen创建本身或smoke当训练完成。未完成的正式评估、延迟/显存成本及科学验收不标PASS。
''',
    'training_health.md':'# 早期训练健康\n\n'+ '\n'.join(f"- {name}: step={v['logged_steps']}, loss={v['last_log']['loss']}, grad_norm(before clip)={v['last_log']['grad_norm']}, elapsed={v['last_log']['elapsed_s']}s, exit={v['exit_code']}" for name,v in snapshot['variants'].items())+'\n\n这是启动快照，不是100k全程健康结论。grad_norm为clip前范数，实际每步clip10；非有限loss/梯度会立即终止并记录非零exit。尚无正式Dev checkpoint成绩。\n',
    'implementation_diff.md':'''# 实现差异

ACTConfig仅新增load_history_steps=0/1/15（默认0关闭），启用时要求6D state；ACT在所有公共参数初始化之后，以隔离CPU RNG/固定seed1建立load MLP和独立零初始化pos embedding。在state之后、image之前追加单token。A1/A2只有输入宽度不同，不进入VAE状态路径，无current/gate/FiLM/残差控制。

新训练runner显式从干净缓存读取6D位置，独立选择single/W15，使用共用Train统计和原图像处理。原录制/部署程序未修改。每组训练模式、预算、seed、数据加载生成器和checkpoint选择一致；公共state_dict hash已实际相同。原P_hist不warmstart。

模型默认关闭回归通过旧8D checkpoint strict加载和完整Dev指标复现。训练入口同时处理原action chunk mask，合法horizon排除padding，h0为主而不使用训练VAE随机latent作为评估预测。

checkpoint保存标准policy/config和f2_1_normalization.json；带load的checkpoint需要本轮runner相同输入适配与历史缓存，不可直接拿旧control_act_det命令上真机。部署适配/真机不在本轮授权。
''',
    'mechanism_review.md':'''# 机制入口已验证，正式结论待训练

GPU smoke验证A1/A2 load层有非零且有限梯度；Dev batch执行原输入、归一化load=0及±0.1标准差扰动。随机初始模型的这种响应只说明计算图连通，不证明学到了有用反馈。

每20k正式Dev评估保存同观测M1/M2逐batch动作变化和样本量，供训练后按帧/episode汇总；A0/A1/A2逐episode主指标一并保存。不用机制诊断替代主指标，不把load变化解释成牛顿力变化。阶段标签仍不新增、不从load推接触。
''',
    'leakage_audit.md':'''# r1 泄漏边界

原始parquet只通过LeRobotDataset明确episodes=Train36或Dev12读取；本地准备也按episode_index过滤。远程整文件SHA256核验是字节校验，不读取Final样本语义。Final列表仅元数据，不进训练、归一化或评估。F1RM Formal5不混入。

state/model/VAE真实6D，current/master不进入batch；A0 batch无observation.load。A1只有load_t，A2按episode内t-14:t首帧左填充。标准化仅Train36，共用均值/标准差，无接触标签。未来action仅作监督，不作为推理latent，评估policy.eval()+zero latent；padding按episode边界排除。

历史8D P_hist只为确认已有主指标流水线，不用于新训练初始化/统计/替代A0。新实验同seed/common hash，固定budget/保存/selection，无后验threshold/窗口/额外variant搜索，无contact/current/residual branch。完整科学与机制结论仍待100k结果。
''',
    'proposal.md':'''# 当前合同内的后续

等待三组100k及每20k自动Dev评估结束，检查exit/complete、每组5个checkpoint、per-episode主指标及M1/M2；按冻结规则选择checkpoint，测推理延迟/显存并生成最终result交Builder和Red-Team验收。

不因训练loss较低提前选赢家，不延长某组、不增seed或FiLM/contact gate/current，不直接改真机部署。若任一训练失败，先保留日志诊断工程原因，不把失败组替换为历史P1。F2.2需要新decision。
'''}
    for name,content in docs.items():
        (OUT/name).write_text(content,encoding='utf-8')
    print(json.dumps(dict(status=overall,variants=snapshot['variants'],remote_code_and_data_hashes_verified=True)))


if __name__=='__main__':main()
