# C50 MASK / INJECT 第三阶段：传播记忆对照与短训划分准备

新端口连接后核对：项目和昨日未提交修改保留，GPU 无既有计算任务。未启动正式训练、未控制机械臂、未提交或推送 GitHub。没有将连接凭证写入文件。

## 可证伪的诊断

1. 若主要是视频传播记忆导致覆盖切换，清空历史后逐帧框提示应减少原异常附近的不稳定。
2. 若单帧框提示也不稳定，清空历史后仍会出现覆盖突变，需要检查提示歧义或单帧分割本身。
3. 若夹爪遮挡歧义是重要因素，异常应集中在遮挡杯体处；后续须用经核查的杯体正点和夹爪负点对照，不能仅凭框外比例确定。

本次仅改变时序策略：每帧 `reset_state` 后用该帧 CVAT cup 框及框外 25px 背景负点调用 SAM2，保留同一 checkpoint、视频、二值阈值和数据。与昨日多关键帧传播作对照。仍允许框外像素，不硬裁框。SAM2 源码核查表明 `reset_state` 清空跟踪结果、对象和提示；保留图像特征缓存不等于保留对象跟踪记忆。

远程输出：`outputs/mask_inject_phase3_20260918/`；screen：`c50_mask_framewise`。

```bash
cd /root/autodl-tmp/lerobot/lerobot-main
export PYTHONPATH=/root/autodl-tmp/c50-mask-env/sam2-minimal:src
/root/autodl-tmp/c50-mask-env/bin/python -u scripts/prepare_c50_mask_preview.py \
  --dataset-root "$PWD/数据集/formal1_C50_nomaster" \
  --output outputs/mask_inject_phase3_20260918/mask_preview_framewise \
  --episodes 0 20 49 --negative-margin 25 --framewise-box \
  --checkpoint checkpoints/sam2.1_hiera_large.pt
```

已有输出拒绝覆盖；复现实验需要新目录。生成成功帧的 `valid=True` 不代表质量认可。

三 episode 全部生成，共 2,154 帧，无空帧。与昨日多关键帧传播相比：

| episode | 平均框外面积占比：传播 → 独立 | >50% 面积突跳：传播 → 独立 |
| --- | --- | --- |
| 0 | 0.120% → 0.106% | 无 → 无 |
| 20 | 1.843% → 0.783% | 400 → 538、541、543、547 |
| 49 | 0.972% → 0.729% | 280、314、332 → 无 |

复查 episode 20 的常规采样、原异常邻帧及新异常邻帧：清空历史后 399–401 不再出现原有的大幅框下溢出，但 537–548 区间仍在“部分可见杯体”与“较完整杯体”覆盖之间切换，夹爪附近出现不同空洞。不能把面积变化阈值直接当成像素准确率。

结论：episode 49 的结果支持传播历史是部分不稳定来源；episode 20 在独立提示下仍出现异常，排除“仅靠清空传播历史即可解决全部问题”。遮挡与提示歧义是下一项待验证假设，不宣称唯一根因已证实。此候选仍未通过质量门槛，不晋升正式训练标签。

下一步用少量人工确认的遮挡关键帧作为像素级核查依据，明确只标可见杯体、排除夹爪，并提供杯体内部正点、夹爪负点；固定这些提示再做传播/独立对照。仅使用框外背景负点不足以约束位于框内的夹爪。不得用检测框中心自动充当杯体正点，因为中心可能被夹爪遮住。低可信帧只能在明确质量审查后标无效，不盲目将所有面积变化帧剔除。

对照量化文件：`corrected_vs_framewise.json`、`baseline_vs_framewise.json`；审查记录：`mask_preview_framewise/review.json`；没有像素真值，本次不报告分割准确率。按 diagnose 技能采用单变量对照，仅新增实验模式，未修改正式标签或动作控制代码。

## 固定短训划分与统计

seed=1000，从既有 train40 中选 development8：`[5,6,8,24,30,33,36,39]`，其余 fit32；已人工查看的 MASK episode 0/20/49 强制留在 fit32。固定 val10 不变，也不称作盲测。

准备清单 `dev_split_with_stats.json` 包含仅由 fit32 的 22,976 帧计算的八维状态和六维动作统计。没有修改当前数据集的 train40 统计或 checkpoint 处理器；短训启动器必须显式使用 fit32 统计，再核对保存的归一化参数。较早的 `dev_split.json` 是无统计的划分清单，后续以 `dev_split_with_stats.json` 为准备依据。

```bash
PYTHONPATH=src /root/miniconda3/bin/python scripts/prepare_c50_dev_split.py \
  --dataset-root "$PWD/数据集/formal1_C50_nomaster" \
  --output outputs/mask_inject_phase3_20260918/dev_split_with_stats.json
```

## 回归与后续门槛

在新端口重新执行：INJECT 6 项、MASK 11 项、时序指标 3 项通过。新增 2 项生成策略测试，用模拟 SAM 输出检查逐帧确实清空状态、默认仍走传播；不以模拟结果证明分割准确率。

未开始 DET/I0/I1/MASK 同预算短训，也未开始最终三组 100k 训练。MASK 标签质量、fit32 处理器创建/检查及实际新三组并发资源验证仍需完成。

## 用户标记后的 SAM 纠正验证

用户提供 ep20 frame538 的红色杯体轮廓与蓝色夹爪标记。提取代表正点 (137,247)、负点 (126,260)/(158,258)，不是将整条轮廓当成正负点，也不是将轮廓当作像素真值。SAM 使用未涂色原图。

单帧保持检测框、框外背景负点不变：加入提示后，原来被掩码包含的一个夹爪负点被排除，两个夹爪负点均在掩码外、杯体正点仍在掩码内；面积 4,058→3,338。右侧绿色掩码完整边界仍待确认，不称作分割准确率。

以 538 为种子向前和向后传播覆盖 533–543：两组对照都无空帧、无 >50% 面积突跳，所以不能将这个局部窗口的稳定性归因于新增提示，不能外推整个 episode。单帧原图再编码与窗口直接抽帧的 JPEG 流程不同，不做跨实验逐元素比较。

结果目录：`sam_prompt_check_ep020_f000538/`、`sam_prompt_neighbors_ep020_533_543/`，均在本阶段目录内。正式标签生产仍未完成，不启动训练；先请用户确认单帧三联图最右侧绿色是否对应其实际杯体。
