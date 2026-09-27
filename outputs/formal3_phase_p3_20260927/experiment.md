# formal3 相位辅助 ACTDet P3 四路训练合同

状态：`AUTHORIZED`。用户已明确授权四路并行100k；脚本仍强制要求远程数据Gate和四路并发2-step资源Gate通过后才启动。

## 问题与单变量

相位辅助监督能否在不改变部署输入/动作接口的前提下，改善透明水杯抓取动作链的阶段表征？本轮只筛选相位loss权重，不回答闭环成功率。

四路：`DET_BASE(use_phase_aux=false)`、`PHASE_W003(0.03)`、`PHASE_W010(0.10)`、`PHASE_W030(0.30)`。其余代码、初始化seed、数据、DET监督、训练预算完全一致。

## 数据合同

- `formal3/kind_merged_nomaster_fit48`；60集/35,917帧物理数据，8D state、6D action、top/gripper 640×480@30FPS。
- seed=1000固定48 train / 12 development：dev=`[9,11,17,25,26,28,30,31,43,52,55,59]`，互斥且不在训练中。
- state/action stats只由48个train episodes计算；图像使用ImageNet stats。
- top cup DET：35,917帧完整；训练仅使用train episodes。gripper DET明确关闭。
- phase：人工reviewed四阶段；unknown=-1不进入loss。不得插值unknown。
- 不含`master_gripper.pos`。

## 训练合同

- 全部从随机初始化开始，seed=1000；ACTDet、chunk=100、n_action_steps=1、gripper loss weight=3。
- DET weight=1；top enabled；MASK、FCOS inject、explicit box、augmentation、AMP均关闭。
- batch=8、num_workers=4、100,000 steps、backbone LR=1e-4、每20k保存、wandb关闭。
- RTX 5090 32GB四路并行；若2-step并发OOM，不静默减batch，停止并修订合同。
- 每路独立输出；任一路失败则总screen返回失败，不覆盖旧输出。中断后先审计checkpoint再决定resume。

## Gate与停止条件

- Gate 1–3：commit一致；数据/标签hash一致；48/12互斥；8D/6D；26项回归通过；真实DET+phase smoke通过。
- 资源Gate：四路batch8同时2步，均exit=0、无OOM/NaN，峰值总显存留有安全余量。
- Training Health：四路100k均done，20k checkpoint齐全，loss有限。
- 本轮结果只能晋升到固定dev12离线阶段评估；不能据此宣称真机改善。
- 停止：任一路NaN/OOM、标签缺失、state schema错误、输出目录已存在或GPU出现其他任务。
