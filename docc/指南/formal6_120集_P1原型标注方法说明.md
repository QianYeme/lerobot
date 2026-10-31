# formal6 共120集数据的 P1 原型标注方法说明

更新日期：2026-10-30。

本文记录历史实际实现，供复现与后续实验规划使用，不启动新的原型生成或训练。

## 1. 名称和范围

这里的“120集”指 formal6 数据集的120条 episode，不是120k训练步数。历史划分为 Train90、Dev15、Final15；本次核查的 P1 训练及杯子原型构建使用 Train90。

当时没有要求用户对120集逐帧画框、描轮廓或点击目标。实际做的是：**自动选出4张训练参考图，由当时的 Codex 查看图像后指定杯内位置，程序据此提取4个 DINO 特征原型。**

因此，准确名称是“少量执行者指定目标位置的原型构建”，不能称为完全无目标先验的自动标注，也不能称为用户完成的人工审核。

## 2. 整体流程

```text
formal6：120集
    │
    ├── Train90 ── 按清单位置固定选4集
    │                  │
    │               各取25%进度处的Top相机帧
    │                  │
    │               Codex查看四帧拼图
    │                  │
    │               指定4个杯内中心坐标
    │                  │
    │               按固定35px采样区域筛选DINO patches
    │                  │
    │               特征均值 + L2归一化
    │                  │
    │               4×384原型库 → P1对象路径
    │
    ├── Dev15：评估与checkpoint选择，不参与原型构建
    └── Final15：封存，不参与原型构建
```

## 3. 第一步：固定选择参考帧

先读取冻结的 Train90 episode 清单，按清单的0%、25%、50%、75%序位各取一集。实际选帧逻辑为：

```python
selected_episodes = [
    train_ids[int(len(train_ids) * q)]
    for q in [0, 0.25, 0.50, 0.75]
]

# 每集选取约25%进度处的一帧，帧号从0开始。
frame_index = int(episode_length * 0.25)
```

历史实际结果如下：

| Episode | 集内帧号，从0开始 | 相机 |
|---:|---:|---|
| 0 | 194 | Top |
| 29 | 194 | Top |
| 58 | 194 | Top |
| 86 | 194 | Top |

这一步按固定规则执行，不是用户挑选图片，也不是根据模型测试成绩挑选参考图。

选帧材料保存在：

- `outputs/p41_prototype_review/candidates.jpg`：四帧拼图。
- `outputs/p41_prototype_review/ep000_f194.jpg` 等：对应单帧。
- `outputs/p41_prototype_review/sources.json`：episode及帧号记录。

## 4. 第二步：Codex查看图片并指定杯内坐标

历史会话记录显示，北京时间2026-10-29：

1. 00:59，Codex提取上述四帧并生成拼图，实际调用了 `view_image` 查看拼图。
2. 随后执行方说明已查看四帧、确认杯子可见。
3. 01:08，Codex在 `scripts/p4_1_pilot.py` 中写入以下固定坐标。

| Episode | 帧号 | x | y |
|---:|---:|---:|---:|
| 0 | 194 | 550 | 325 |
| 29 | 194 | 448 | 279 |
| 58 | 194 | 421 | 251 |
| 86 | 194 | 505 | 300 |

坐标对应原始 Top 相机图像：宽640、高480；左上角为原点，x向右增加，y向下增加。历史构建脚本没有在读帧后执行翻转；不能将后来LIBERO的图像翻转规则套到这里。

这些坐标代表执行方指定的杯内采样位置，不是经过验证的精确杯中心、抓取点、杯口中心或水面中心。没有发现目标检测器自动生成这四个坐标的代码，也没有发现用户点击、确认这些坐标的记录。

保存的来源说明使用了 `executor reviewed Train images`。其中的 executor 指当时执行任务的 Codex；这不能作为独立人工审核通过的证明。

## 5. 第三步：用坐标选择特征采样区域

实际函数为 `scripts/p4_1_pilot.py` 中的 `prototypes(ids)`。其核心参数是：

```python
sources = [
    (0, 194, (550, 325)),
    (29, 194, (448, 279)),
    (58, 194, (421, 251)),
    (86, 194, (505, 300)),
]
```

程序读取对应原始RGB帧，以冻结的 DINOv2-S/14 提取特征：

1. RGB转为浮点 `[0, 1]`。
2. 在DINO封装内双线性缩放至224×224。
3. 执行一次ImageNet标准化。
4. 得到16×16个patch，每个patch为384维特征。

随后将16×16网格中心映射回640×480原图，按下面的规则选择patch：

```python
yy, xx = torch.meshgrid(
    torch.arange(16, device="cuda"),
    torch.arange(16, device="cuda"),
    indexing="ij",
)

mask = (
    (((xx + 0.5) * 640 / 16 - center_x) / 35).square()
    + (((yy + 0.5) * 480 / 16 - center_y) / 35).square()
) <= 1

assert mask.sum() > 0
```

这里沿x、y方向的采样半径均为原图35像素，直径约70像素。判断的是**patch中心是否位于采样区内**，不是逐像素描出杯子的轮廓；实际特征仍对应完整patch，不能保证其中没有背景、夹爪或其它上下文。

这个 `mask` 只是特征采样用的布尔网格，不是SAM分割结果，也不是用于训练分割头的真值标签。

## 6. 第四步：生成并保存4个原型

对每张参考图被选中的patch特征取均值，再执行L2归一化：

```python
prototype = torch.nn.functional.normalize(
    patches[0, mask.flatten()].mean(dim=0),
    dim=0,
)

bank.append(prototype.cpu())

# 四张图各产生一个384维向量。
prototypes = torch.stack(bank)  # [4, 384]
```

历史产物位于：

```text
experiments/p4_final_p1_training/
  p4_1_unattended_p1_pilot/index3/result/
    formal6_train90_prototypes.pt
    prototype_provenance.json
```

保存内容包括原型张量，以及每个原型对应的episode、帧号、中心坐标、原始RGB像素SHA256和采样说明。

已核查的原型库文件SHA256为：

```text
787a7084f1a387c71caaae635618458ec620b750d258df5df52578b1e7bc272f
```

本次只核验历史文件、代码和来源，不重新生成或覆盖它。不同序列化环境重新生成的文件不必然具有相同文件哈希；复现时还应分别比较来源和张量数值。

## 7. 这些“标注”如何参与P1训练

四个坐标只用于准备原型库，不是每个训练样本都额外输入四个坐标。

P1处理新的Top图像时，提取DINO patch特征，计算各patch与四个原型的余弦相似度，并取最高相似度。模型据此进行温度为0.20的软聚合，产生1个全局和4个固定2×2局部object token，再通过adapter及带门控的cross-attention影响ACT编码器输入。

因此，这些原型的作用是提供“应关注哪一类视觉特征”的参考。它们不提供逐帧目标真值，也不保证模型一定准确定位或抓到杯子。

原型构建侧始终是单次ImageNet标准化。历史Legacy和后来Fixed在线DINO输入的标准化差异是另一个实现问题，不能与原型坐标来源混为一谈。

## 8. 复用这套方法时应怎样描述

建议描述：

> 从冻结训练划分按固定规则抽取4张参考图，由执行方查看图像并指定目标内部采样中心。利用固定局部采样规则提取冻结DINOv2-S特征，经均值与L2归一化构成4个目标参考原型。没有逐帧目标标注，但原型生成包含少量执行者指定的目标位置先验。

迁移到新数据集时可以复用选帧、特征提取和保存机制，但必须重新确定目标位置，不能复制本表的杯子坐标或直接复用杯子原型。

如果新实验禁止执行者看图指定目标位置，那么这套历史方法不满足其无标注约束。本文仅说明历史做法，不修改当前P4.8-r3的科研授权。

## 9. 证据索引及局限

- 构建代码：`scripts/p4_1_pilot.py`，`prototypes(ids)`。
- 原始坐标及像素来源：上述 `prototype_provenance.json`。
- 实际查看图像和写入代码的历史补证：`experiments/p4_public_benchmark/p4_8_libero_pilot/index3/revisions/r2/result/prototype_history_followup.md`。
- 原脚本及服务器缓存核验：同一归档目录的 `source_audit_raw.json`、`historical_train_membership.json`。

目前没有独立人工审核记录或点击轨迹；查看图像并写入坐标不能证明区域选取精确。能确认的是固定选帧、Codex实际查看图片、后续固定坐标入代码，以及保存原型文件的来源记录相互对应。
