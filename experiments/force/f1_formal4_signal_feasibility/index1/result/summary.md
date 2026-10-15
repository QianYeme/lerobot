# F1 r1：信号工程预检交付，独立标签门待完成

## 最新进展：前六集人工标注已核验

用户确认完成标注；实际CSV有30条区间、审核者lyj。服务同一验证器已检查标签、起止帧、可靠/不确定区间及同轴冲突，原CSV保持不变；原件快照/哈希及旧报告版本保存在独立label_gate raw目录，见first_six_label_gate.json。

W15合法历史后PRECONTACT/CONTACT_SUPPORTED可靠目标帧数：Train0=186/189、2=138/180、4=163/215、6=164/164、7=161/168、9=145/184。**6/6达到每类≥5帧，通过≥4/6门，按合同继续Train36接触任务审核，不降级阶段。**

目前仅审核前6集，剩余30集Train未审；全Train需至少24/36集两类覆盖达标后才能冻结主任务和标签版本，再审核Dev。现在不能拟合正式probe、报告BA/CI或解释load是否有增量。六集UI尚未扩展；旧“尚未审核前六集”的描述仅是下方预检历史，已由本段及最新label_coverage/task_freeze替代。人工区间不代表Codex完成连续播放复核。

OVERALL_STATUS = PARTIAL_PASS  
F1_SIGNAL_FEASIBILITY_COMPLETE = false  
SCIENTIFIC_OUTCOME = UNRESOLVED  
LOAD_PRIMARY_SUPPORTED = false  
SUPPORT_SOURCE = NONE  
NEXT_POLICY_TRAINING_AUTHORIZED = false

## 已实际完成

- 启动时未发现旧F1 result/raw或r0科研指标；本轮使用用户指定r1文件，不改合同。实际代码分支det、HEAD/dirty diff/依赖及合同、split、源码哈希已保存provenance.json和独立raw run目录。
- 按冻结Train36/Dev12过滤共享parquet，未读取Final数据行或视频。核对17,243/5,748帧、8D状态/6D动作、连续索引与30FPS时间戳。
- 逐split、逐episode信号range、mean/std、分位数、零值率、唯一值/观测量化、非有限计数和首尾变化已写signal_integrity.json。load/current均非训练集常量，检查到的三个信号均无非有限值。原生寄存器读数未标定为力；样本极值不当作饱和阈值。
- 当前录制源码先观测再发送同一行action，S0仅用t−1及更早发送目标。已实现W=1/9/15固定模板，共同有效掩码和逐episode历史缓存，不使用时间/episode作为输入。每集窗口排除计数见leakage_audit.json。
- 3项CPU定向测试实际通过：单帧模板/差分、同一行及未来命令不可影响过去特征、完整窗口/统一非有限掩码。没有拟合scaler或probe，更未训练policy。
- 冻结Train顺序前六集0/2/4/6/7/9完整双视角片段、30张定位图和盲审表已随包准备；副本SHA256和帧数核对。没有借用Formal3标签或把D2.1姿态观察充作F1接触标签。

## 阻塞与尚未执行

当前工具提供图片查看而非原生连续视频观看。D2.1曾按有序连续帧联系表审核，但这不满足本合同明确要求的“连续播放并逐帧核查”；本轮不再以粗抽帧或全UNKNOWN占位表冒充完成。

前六Train独立视频审核尚未执行，因此不能判断4/6接触覆盖门、不能擅自降级到阶段任务；Train任务未冻结，Dev标签审核也未启动。metrics/bootstrap/confusion/matching均明确NOT_RUN，无BA、F1、delta、CI或支持率可报告。窗口/probe/逐集delta/CI图待标签门通过才生成，当前仅有信号分布图。

历史采集代码/设备标定/固件身份、视频曝光与遥测硬件时间同步尚未独立核验；当前源码可定义保守因果命令历史，但不等于完整历史部署等价。D2/D2.1未闭环、G1正式真机评价未完成、B3为既有OC-ACT候选的事实不变。

## 唯一下一步

现已提供本地辅助页面：运行 `python scripts/run_f1_review_ui.py`，打开 http://127.0.0.1:8766 。可逐帧取值、多段区间并直接保存原审核CSV（备份/回读/冲突保护）；使用说明见项目tools/f1_review_ui/README.md，18项回归及真实Edge隔离CSV测试结果见ui_validation.json。工程测试没有产生科研标签，不代表F1完成。

按[evidence_samples/REVIEW_FIRST_SIX.md](evidence_samples/REVIEW_FIRST_SIX.md)完成6集RGB盲审，只填写可靠区间和不确定边界。随后继续本合同确定的接触→阶段覆盖分支，再冻结任务、审核Dev、拟合固定逻辑回归。无需新真机、服务器GPU或新数据。

这不是load无用或力控失败的结论，LOAD_PRIMARY_SUPPORTED=false仅表示尚未取得支持证据。没有进入F2、改架构/部署、训练策略、读取Final或push。
