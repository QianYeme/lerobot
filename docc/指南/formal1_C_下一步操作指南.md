# formal1_C：静态杯位接近实验操作指南

日期：2026-09-15。适用于当前项目 `det` 分支。

## 当前优先执行：前 50 集临时训练方案

后 40 集（episode 050–089）的现有检测标注与 `formal1/kind2` 视频不匹配，正在重新标注。当前先使用已经人工检查通过的 episode 000–049 完成测试和训练。下面这份临时方案优先于后文的 90 集方案；在后 40 集重新验收通过之前，不执行原来的 72/18 划分和 90 集训练命令。

本轮仍只验收静态杯位接近：动作应随杯位改变，并由操作者在接触前停止。训练使用完整演示，不提前裁掉抓取和放下阶段。

### A. 数据选择：40 集训练、10 集验证

继续使用现有 `formal1_C` 目录，不需要重新切分、拼接视频或复制数据。通过 episode 清单限制实际采样范围，禁止后 40 集进入本轮训练和验证。

先按后文第 3 节设置 PROJECT、DATA_ROOT 并准备环境，再在执行训练/评估的 Bash 会话中运行以下命令，替代第 4 节的 72/18 清单：

```bash
export VAL_EPISODES='7,11,13,14,16,27,35,38,40,43'
export TRAIN_EPISODES="$(python - <<'PY'
import json
val = {7,11,13,14,16,27,35,38,40,43}
print(json.dumps([ep for ep in range(50) if ep not in val]))
PY
)"

mkdir -p outputs/formal1_C50_phase1
python - <<'PY'
import json, os
from pathlib import Path
train = json.loads(os.environ['TRAIN_EPISODES'])
val = list(map(int, os.environ['VAL_EPISODES'].split(',')))
assert len(set(train)) == 40 and len(set(val)) == 10
assert not set(train) & set(val)
assert set(train) | set(val) == set(range(50))
Path('outputs/formal1_C50_phase1/split.json').write_text(
    json.dumps({'train': train, 'validation': val}, indent=2))
print('Split OK: 40 train / 10 validation, episodes 0-49 only')
PY
```

| 方位 | 前 50 集总数 | 训练 | 验证 |
|---|---:|---:|---:|
| 左侧 | 8 | 6 | 2 |
| 左前 | 15 | 12 | 3 |
| 正前 | 3 | 2 | 1 |
| 右前 | 15 | 13 | 2 |
| 右侧 | 9 | 7 | 2 |
| 合计 | 50 | 40 | 10 |

每次训练必须传入 `--dataset.episodes="$TRAIN_EPISODES"`，评估必须传入 `--episodes "$VAL_EPISODES"`。meta/info.json 中的 train 仍然是 0:90，不能靠它自动排除错误标注。

### B. 前 50 集验收与训练

第 5 节的叠图检查如果重跑，将代码中的 `for ep in range(90):` 改为 `for ep in range(50):`。最后的提示文字也改成 50 episodes；其他全数据结构检查可以保留，因为底层仍是完整 formal1_C。这样会生成前 50 集共 500 张检查图片。

本轮训练三个模型，使用仓库脚本 `scripts/train_c50.sh`。不再依赖第 6 节的临时 `train_c` 函数，新开 screen 会话也能直接运行。

| 模型 | 脚本参数 | 正式实验名 | 检测 / 注入 |
|---|---|---|---|
| ACT 基准 | act | C50_ACT_s1000 | 无检测 |
| ACTDet 检测 | det | C50_DET_s1000 | top 检测，额外特征注入关闭 |
| ACTDet 检测注入 | inject | C50_INJECT_s1000 | top 检测，开启 FCOS p4 特征注入 |

三组都使用两路图像进行动作预测，关闭 Mask 和在线增强；seed=1000、batch=8、chunk_size=100、夹爪权重=3.0、视觉主干学习率=1e-4。检测与注入组之间只改变 `fcos_feature_inject`。`p4` 对应当前实现支持的注入层。

#### B1. 启动前准备

先将本地更新的指南及 `scripts/train_c50.sh` 同步到服务器。脚本尚未到服务器时不能执行以下命令。已安装依赖的 Python 环境需要在启动 screen 前激活。

```bash
cd /root/autodl-tmp/lerobot/lerobot-main
export DATA_ROOT="$PWD/数据集/formal1_C"
export PYTHONNOUSERSITE=1
command -v screen
command -v python
command -v lerobot-train
test -f scripts/train_c50.sh
bash -n scripts/train_c50.sh
screen -ls
```

如果缺少 screen，在该 Ubuntu 容器中安装后再继续：

```bash
apt-get update && apt-get install -y screen
```

脚本自动根据自身位置找到项目、默认使用项目下的数据集，并在每次启动时生成固定的 40/10 清单；不依赖终端里残留的 TRAIN_EPISODES。DATA_ROOT 可用于指定另一份同内容数据的位置。第 A 节的变量仍供手动评估使用。

#### B2. 三组冒烟训练：各 2,000 步

推荐单 GPU 串行执行：

```bash
screen -S C50_smoke bash scripts/train_c50.sh all 2000
```

脚本按 ACT → 检测 → 检测注入运行；任一组失败就停止队列。各自实验名为 `C50_ACT_smoke_s1000`、`C50_DET_smoke_s1000`、`C50_INJECT_smoke_s1000`。

启动后按 **Ctrl+A，再按 D** 脱离 screen，训练继续。SSH 断开后重新登录，用以下命令返回：

```bash
screen -ls
screen -r C50_smoke
```

如果同名会话仍显示 Attached，先确认没有另一人在使用；需要转移自己的旧连接时使用 `screen -d -r C50_smoke`。脚本完成或报错后该 screen 会话会结束，应查看持久化日志，不能仅凭会话消失判定成功。

已完成的 ACT 冒烟不必覆盖重跑。已有输出目录时脚本会退出，保留原结果；只启动尚未完成的组，例如：

```bash
screen -S C50_det_smoke bash scripts/train_c50.sh det 2000
# 上一组结束并通过后再执行：
screen -S C50_inject_smoke bash scripts/train_c50.sh inject 2000
```

#### B3. 正式首轮：三组各 20,000 步

确认三组冒烟均无报错、checkpoint 已保存且检测/注入组有有效检测损失后，执行：

```bash
screen -S C50_train bash scripts/train_c50.sh all 20000
```

依次运行三个实验，单卡默认采用这种方式。20,000 步是首次比较预算，不是收敛保证；冒烟和正式训练使用不同目录，正式训练重新初始化策略，不接续冒烟权重。

若要单独管理某个模型，可用以下三个独立会话命令。单 GPU 请每组结束后再启动下一组；不要在串行队列运行期间再次启动同一模型。

```bash
screen -S C50_ACT bash scripts/train_c50.sh act 100000
screen -S C50_DET bash scripts/train_c50.sh det 100000
screen -S C50_INJECT bash scripts/train_c50.sh inject 100000
```

只有确认多 GPU 可用时才分卡并行，例如 `CUDA_VISIBLE_DEVICES=1 screen -S C50_DET bash scripts/train_c50.sh det 20000`。`screen` 本身不分配 GPU，也不隔离显存。

#### B4. 日志、恢复与检查

```bash
# 不进入 screen 也能查看进度；Ctrl+C 只退出 tail，不会停止训练。
tail -f outputs/formal1_C50_phase1/C50_ACT_s1000/train.log
tail -f outputs/formal1_C50_phase1/C50_DET_s1000/train.log
tail -f outputs/formal1_C50_phase1/C50_INJECT_s1000/train.log
```

每组日志目录还保存 `command.sh`、`split.json`、`code_commit.txt`、已跟踪代码修改补丁及环境清单。checkpoint 保存在 `outputs/train/<实验名>/checkpoints/`。如果使用之前临时函数完成了某组实验，旧日志仍在 `outputs/formal1_C_phase1/`，不会被脚本搬动。

重新执行同名脚本不会覆盖或自动恢复已有实验。中断后通过 checkpoint 恢复，例如：

```bash
screen -S C50_DET_resume bash -c 'set -o pipefail; lerobot-train --config_path=outputs/train/C50_DET_s1000/checkpoints/last/pretrained_model/train_config.json --resume=true --steps=20000 2>&1 | tee -a outputs/formal1_C50_phase1/C50_DET_s1000/resume.log'
```

从中断处继续 20k 就用 `--steps=20000`；根据验证结果决定延长到 60k 时用 `--steps=60000`。ACT/注入组替换路径与会话名。恢复前确认配置文件存在；若尚未产生 checkpoint，不能恢复，先检查失败原因并另行保留/处理失败目录。

启动日志核对：训练恰好 40 集、编号全部小于 50；两检测组 Mask=false；检测组注入=false，注入组注入=true、levels=[p4]。三组保持相同训练预算；显存不足时统一调整 batch 并记录，不只改其中一组。

### C. 离线与真机评估如何替换

第 7 节的评估循环首行改为：

```bash
for RUN in C50_ACT_s1000 C50_DET_s1000 C50_INJECT_s1000; do
```

其余循环体沿用第 7 节，并确保当前 VAL_EPISODES 仍是上面的 10 集。按方位评估时使用：左侧 `14,38`；左前 `13,27,43`；正前 `7`；右前 `16,40`；右侧 `11,35`，不使用后文包含 50 以上编号的五组清单。

真机测试仍按第 9–10 节执行，但 CKPT 改成选中的 C50 checkpoint，TRIAL 名称加上 C50 标识。若延长训练或恢复 checkpoint，第 6 节恢复命令中的实验名也替换为对应 C50 名称。

### D. 本轮能回答什么，以及限制

- 可以验证检测监督是否参与学习、零 z 推理表现是否改善、动作是否随杯位改变，以及真机接近流程是否可用。
- 正前方只有 2 集训练和 1 集验证，单次结果不稳定，不能据此宣称该方向已经可靠泛化。
- 当前仍沿用 formal1_C 全部 90 集的归一化统计；限制 episodes 不等于重新计算训练 40 集统计。作为阶段性排查需注明这一点，正式严格对照时应使用仅训练集计算的统计。
- 前 50 集训练是阶段性基线，不能替代最终 90 集实验，也不要与 90 集结果混用实验名称。

后 40 集重新标注完成后，先检查两路相机、逐集边界及动态阶段叠图，再固定新的数据版本，恢复后文的 72/18 划分。正式比较建议三模型按统一设置重新训练；若从 C50 模型继续微调，应另记为微调实验。下文第 1–12 节保留原 90 集两模型方案作为参考，当前三模型训练以本节为准，不能用 C50 脚本直接启动 90 集实验。

---

## 1. 本轮目标与范围

已确认的目标：杯子在每次开始时处于不同的静态位置，机械臂从基本相同的初始关节姿态出发，运动方向与接近终点随杯位正确改变，在接触杯子前由操作者停止。

本轮不考核夹爪闭合、拿起、放下，也不要求运行中移动杯子后的动态追踪。训练仍使用完整演示；“接触前停止”是评测协议，不代表训练数据已经截断或控制器具备自动停止能力。

执行顺序：数据下载与版本记录 → 标注叠图验收 → 固定训练/验证清单 → 两组短训练 → 离线评估 → 五方位真机接近 → 根据失败证据决定下一项改动。

本指南是下一轮实验方案，不代表这里的服务器命令、训练结果和真机通过率已经实测。命令参数已按当前源码核对。下文数值门槛是本项目的初始验收建议，不是通用标准。

## 2. 数据和实验约定

数据集：[QYyyyyyyy/formal1_C](https://huggingface.co/datasets/QYyyyyyyy/formal1_C)。

| 项目 | 当前内容 |
|---|---|
| 规模 | 90 集、64,591 帧、30 FPS |
| 000–049 | 新 50 集，轨迹/视频来自 formal1/kind，XML 来自 fk |
| 050–089 | 原随机 40 集，轨迹/视频来自 kind2，XML 来自 formal1_B |
| 图像 | top / gripper，640×480，原始 AV1 |
| 视频存储 | 每路两个文件，由 episode 元数据定位，不需要拼接 |
| 状态 / 动作 | 9 维 / 6 维，顺序见 meta/info.json |
| 标注 | top 和 gripper 各 90 个 CVAT XML，类别 cup |
| Mask | 未生成完整 90 集 Mask，本轮关闭 |

第一轮两个实验：

| 名称 | 策略 | 检测监督 | 其他设置 |
|---|---|---|---|
| C_ACT_s1000 | 纯 ACT | 无 | 两路图像、完整动作监督 |
| C_DET_s1000 | ACTDet | 仅 top | 关闭 Mask、额外特征注入及在线增强 |

两个模型使用相同训练集、seed、batch、步数和夹爪权重。先关闭增强以减少对照中的变量；后续需要时再单独开启已修复的颜色增强。gripper 图像仍参与动作预测，只是首轮不启用其检测监督。

保留标准 CVAE 设置，推理时用零 z。先用数据证据判断视觉是否有效，不把“CVAE 压缩必然导致不泛化”作为结论。

旧工作日志中的“先冻结检测器再训练策略”是候选路线。当前检测器与策略共享视觉主干，本轮先使用现成的联合训练流程做对照；只有感知确实成为瓶颈时，再设计检测预训练、冻结范围和权重迁移。

旧 RUN_GUIDE 的部分结论已经过时：任务描述拆成两类不能自动保证 ACT 随杯位运动；不同损失的数值大小也不能直接代表共享主干的梯度贡献。不要照搬旧数据路径或一次重训全部旧实验。

## 3. 服务器准备与数据下载

下列命令使用 Linux Bash，假设项目位于 `/root/autodl-tmp/lerobot/lerobot-main`，且已激活安装本项目的 Python 环境。若路径不同，仅修改 PROJECT。不要把这些 Bash 命令直接粘贴到 Windows PowerShell。

```bash
export PROJECT=/root/autodl-tmp/lerobot/lerobot-main
export DATA_ROOT="$PROJECT/数据集/formal1_C"
export PYTHONNOUSERSITE=1
cd "$PROJECT"
git status --short --branch
```

工作区干净时同步代码；如果有自己的未提交修改，先保留修改并处理分支状态，不强制覆盖。

```bash
git fetch origin
git switch det
git pull --ff-only origin det
git merge-base --is-ancestor 1f72cd03 HEAD
```

最后一条应退出 0，表示包含零 z 离线评估、视觉敏感性和增强修复。若提交不可见，先核对服务器仓库地址与分支。

```bash
python -c "import sys,torch,av,lerobot; print(sys.executable); print(lerobot.__file__); print(torch.__version__,torch.cuda.is_available(),av.__version__)"
mkdir -p outputs/formal1_C_phase1
git rev-parse HEAD > outputs/formal1_C_phase1/code_commit.txt
python -m pip freeze > outputs/formal1_C_phase1/environment.txt
```

CUDA 应可用，lerobot 路径应指向这份项目。缺少依赖时先恢复环境，不直接开始长训练。以下通过已安装的 Hugging Face Python API 下载公开数据，不需要写 Token：

```bash
python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
repo = 'QYyyyyyyy/formal1_C'
revision = HfApi().repo_info(repo, repo_type='dataset').sha
snapshot_download(repo_id=repo, repo_type='dataset', revision=revision,
                  local_dir=os.environ['DATA_ROOT'])
Path('outputs/formal1_C_phase1/dataset_revision.txt').write_text(revision + '\n')
print('Dataset revision:', revision)
PY
```

文件夹应直接包含 meta/data/videos/annotations。不要再嵌套一层 formal1_C。下载工具可能生成 .cache 和 .gitattributes，不属于训练样本。

## 4. 固定 72/18 集划分

以下清单根据已有两个首帧方位 CSV 生成：在每个“来源×方位”组内按 episode 排序，取分散的验证样本。新 50 集留出 10，随机 40 集留出 8。原 kind2 编号在合并集中加 50。

| 方位 | 全部 | 训练 | 验证 | 验证 episode |
|---|---:|---:|---:|---|
| 左侧 | 14 | 11 | 3 | 14,38,86 |
| 左前 | 23 | 18 | 5 | 13,27,43,80,84 |
| 正前 | 13 | 10 | 3 | 7,55,77 |
| 右前 | 25 | 21 | 4 | 16,40,52,72 |
| 右侧 | 15 | 12 | 3 | 11,35,65 |
| 合计 | 90 | 72 | 18 | |

在执行训练和评估的 Bash 会话中设置：

```bash
export TRAIN_EPISODES='[0,1,2,3,4,5,6,8,9,10,12,15,17,18,19,20,21,22,23,24,25,26,28,29,30,31,32,33,34,36,37,39,41,42,44,45,46,47,48,49,50,51,53,54,56,57,58,59,60,61,62,63,64,66,67,68,69,70,71,73,74,75,76,78,79,81,82,83,85,87,88,89]'
export VAL_EPISODES='7,11,13,14,16,27,35,38,40,43,52,55,65,72,77,80,84,86'
python - <<'PY'
import json, os
from pathlib import Path
train = json.loads(os.environ['TRAIN_EPISODES'])
val = list(map(int, os.environ['VAL_EPISODES'].split(',')))
assert len(set(train)) == 72 and len(set(val)) == 18
assert not set(train) & set(val)
assert set(train) | set(val) == set(range(90))
Path('outputs/formal1_C_phase1/split.json').write_text(
    json.dumps({'train': train, 'validation': val}, indent=2))
print('Split OK: 72 train / 18 validation')
PY
```

数据集 meta/info.json 的 train 仍是 0:90；实际训练依赖显式传入 `--dataset.episodes`，不能遗漏。不要重编号验证集，也不要把相邻帧随机拆到两边。

这是同一采集环境内的验证集，不是跨环境测试集。相似杯位、相近录制条件仍可能让结果偏乐观。当前流程沿用数据集已有归一化统计，没有单独重算训练 72 集的状态/动作统计，因此不能称为完全无统计泄漏的独立测试；若用于正式论文，应另外落实仅训练集统计和独立测试集。

## 5. 训练前的数据与标注验收

先检查全局索引，再通过 LeRobot 实际读取样本并叠加 XML。下面的脚本只生成检查图片，不改变数据集。

```bash
python - <<'PY'
import os, json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw
from lerobot.datasets.lerobot_dataset import LeRobotDataset

root = Path(os.environ['DATA_ROOT'])
info = json.loads((root/'meta/info.json').read_text())
assert info['total_episodes'] == 90 and info['total_frames'] == 64591
assert info['fps'] == 30
table = pq.read_table(root/'data/chunk-000/file-000.parquet')
indices = np.asarray(table['index'].to_pylist())
assert np.array_equal(indices, np.arange(64591))
eps = np.asarray(table['episode_index'].to_pylist())
frames = np.asarray(table['frame_index'].to_pylist())
ds = LeRobotDataset('QYyyyyyyy/formal1_C', root=root, video_backend='pyav',
                   delta_timestamps={'action': [i/30 for i in range(100)]})
out = Path('outputs/formal1_C_phase1/annotation_review')
out.mkdir(parents=True, exist_ok=True)
for ep in range(90):
    ids = np.flatnonzero(eps == ep)
    assert np.array_equal(frames[ids], np.arange(len(ids)))
    annotations = {}
    for cam in ('top', 'gripper'):
        tree = ET.parse(root/f'annotations/{cam}/episode_{ep:03d}.xml')
        boxes = {}
        for track in tree.getroot().findall('track'):
            assert track.get('label') == 'cup'
            for box in track.findall('box'):
                f = int(box.get('frame'))
                assert 0 <= f < len(ids)
                if box.get('outside', '0') != '1':
                    xy = tuple(float(box.get(k)) for k in ('xtl','ytl','xbr','ybr'))
                    assert 0 <= xy[0] < xy[2] <= 640
                    assert 0 <= xy[1] < xy[3] <= 480
                    boxes.setdefault(f, []).append(xy)
        annotations[cam] = boxes
    for f in sorted({0, len(ids)//4, len(ids)//2, 3*len(ids)//4, len(ids)-1}):
        sample = ds[int(ids[f])]
        assert int(sample['episode_index'].item()) == ep
        assert int(sample['frame_index'].item()) == f
        if f == 0:
            assert not sample['action_is_pad'].any().item()
        for cam in ('top', 'gripper'):
            rgb = sample[f'observation.images.{cam}']
            assert tuple(rgb.shape) == (3,480,640)
            im = Image.fromarray((rgb.permute(1,2,0).numpy()*255).clip(0,255).astype('uint8'))
            draw = ImageDraw.Draw(im)
            boxes = annotations[cam].get(f, [])
            for xy in boxes:
                draw.rectangle(xy, outline='red', width=3)
            draw.text((8,8), f'{cam} ep={ep:03d} frame={f} boxes={len(boxes)}', fill='red')
            im.save(out/f'{cam}_ep{ep:03d}_f{f:04d}.png')
print('OK: 90 episodes checked; review images:', out)
PY
```

该检查按已有 CVAT track/box 格式读取。如果断言失败，先记录具体 episode/frame 并检查原因，不删除断言来绕过。脚本检查的是 XML 显式保存的框，不额外推断缺失帧的插值标注。

人工查看方法：

1. 先看 90 集 top 首帧，再看 gripper 首帧，确认相机语义和杯位。
2. 检查所有采样阶段图片，重点看 49→50 的来源切换、左右极端位置、遮挡和透明杯反光。
3. 对疑似错位的集查看相邻连续帧：框应跟随杯子，不应框在机械臂、杯影或别的反光物上。只看首帧不足以排除后半段偏移。
4. 区分“杯子不可见且无框”和“杯子可见但漏标”。部分旧 XML 起始几帧没有框，不能直接作为真实负样本接受；逐集检查，必要时补标并重新固定数据版本。

通过条件：没有相机反置、系统性帧偏移、可见杯子的成段漏标或明显错误框。截图文件保留为验收证据。抽样通过不代表每帧标注都正确。

## 6. 两组训练：先 2,000 步冒烟，再 20,000 步试验

先在当前 Bash 会话定义训练函数。输出日志放在实验目录外，避免训练器因 output_dir 已存在而拒绝启动。

```bash
set -o pipefail
train_c () {
  local policy_type="$1" run_name="$2" train_steps="$3"
  local extra=()
  if [ "$policy_type" = act_det ]; then
    extra=(
      --policy.use_detection=true
      --policy.use_mask_guidance=false
      --policy.fcos_feature_inject=false
      --policy.mask_feature_inject=false
      --policy.aug_enable=false
      --policy.det_weight=1.0
      --policy.annotation_dir="$DATA_ROOT/annotations"
      '--policy.det_cameras={"observation.images.top":{"enable":true},"observation.images.gripper":{"enable":false}}'
    )
  fi
  lerobot-train \
    --policy.type="$policy_type" \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --dataset.repo_id=QYyyyyyyy/formal1_C \
    --dataset.root="$DATA_ROOT" \
    --dataset.episodes="$TRAIN_EPISODES" \
    --dataset.video_backend=pyav \
    --dataset.use_imagenet_stats=true \
    --dataset.image_transforms.enable=false \
    --policy.chunk_size=100 \
    --policy.n_action_steps=1 \
    --policy.gripper_loss_weight=3.0 \
    --policy.optimizer_lr_backbone=0.0001 \
    --seed=1000 --batch_size=8 --num_workers=4 \
    --steps="$train_steps" --log_freq=100 --save_freq=2000 \
    --eval_freq=0 --wandb.enable=false \
    --output_dir="outputs/train/$run_name" \
    "${extra[@]}" 2>&1 | tee "outputs/formal1_C_phase1/$run_name.log"
}
```

```bash
train_c act C_ACT_smoke_s1000 2000
train_c act_det C_DET_smoke_s1000 2000
```

检查：能保存 checkpoint；损失无 NaN/Inf；ACTDet 日志包含检测损失，不能只有动作损失；有正框的 batch 应产生有效回归监督。纯 ACT 没有检测损失是正常的。2,000 步只验证训练链路，不判断最终跟踪能力。

显存不足时两个实验都改成 batch=4，并以新实验名重跑，记录变化；DataLoader 问题可先改 num_workers=0 定位。不要只对一组悄悄改变训练预算。

冒烟通过后启动新的 20,000 步试验。它们从相同初始化规则重新训练，不加载旧损坏数据训练出的策略权重；ImageNet 主干预训练保留。

```bash
train_c act C_ACT_s1000 20000
train_c act_det C_DET_s1000 20000
```

使用已有的 screen/tmux 会话运行长任务；新开会话要重新设置环境变量和函数。20,000 步不是收敛保证，不因该点效果差就否定架构。先比较 10k/20k 验证结果，再决定是否延长。

`eval_freq=0` 关闭训练器的环境评估；它不会自动对本文 18 集做离线验证。下一节必须另外运行。

若需要从该试验最后的 checkpoint 延长到 60k，而不是重新初始化：

```bash
lerobot-train \
  --config_path=outputs/train/C_DET_s1000/checkpoints/last/pretrained_model/train_config.json \
  --resume=true --steps=60000
```

先确认该路径存在，并在启动日志中核对恢复步数、目标步数、训练 72 集和 Mask=false。ACT 同理替换实验名。新服务器恢复时，配置中的绝对数据和标注路径也必须有效。

## 7. 离线动作评估与视觉敏感性

先对两组 last 运行，后续同样检查 10k/20k 等候选。checkpoint 路径必须指向含 config.json/model.safetensors/处理器文件的 pretrained_model。

```bash
for RUN in C_ACT_s1000 C_DET_s1000; do
  CKPT="outputs/train/$RUN/checkpoints/last/pretrained_model"
  python -m lerobot.scripts.offline_eval_act_det \
    --checkpoint "$CKPT" \
    --dataset.repo_id QYyyyyyyy/formal1_C --dataset.root "$DATA_ROOT" \
    --episodes "$VAL_EPISODES" --dataset.video_backend pyav \
    --annotation-dir "$DATA_ROOT/annotations" \
    --batch-size 8 --num-workers 0 \
    --output "outputs/formal1_C_phase1/${RUN}_offline.json"

  python -m lerobot.scripts.eval_visual_sensitivity \
    --checkpoint "$CKPT" \
    --dataset.repo_id QYyyyyyyy/formal1_C --dataset.root "$DATA_ROOT" \
    --episodes "$VAL_EPISODES" --dataset.video_backend pyav \
    --camera-key observation.images.top \
    --batch-size 8 --num-workers 0 --max-batches 100 \
    --action-steps 10 --seed 1000 \
    --output "outputs/formal1_C_phase1/${RUN}_visual.json"
done
```

判读方法：

| 观察 | 可以说明什么 | 不能直接说明什么 |
|---|---|---|
| inference_l1_loss 下降 | 零 z 推理在留出集更接近演示动作 | 真机一定成功 |
| swap_action_delta_l1 明显非零 | 换 top 图会改变动作 | 改变方向一定正确 |
| swap_error_increase 为正 | 错配图像通常使动作预测变差 | 已经证明利用的是杯子而非背景 |
| ACTDet det loss 下降 | 检测监督参与训练且拟合改善 | 检测框定位准确、召回率合格 |

现有换图脚本从整集随机采帧，换 top 时保留原状态和 gripper，会制造跨阶段、跨相机不一致。它是诊断工具，不是严格的杯位因果实验，也不是仅接近阶段指标。全动作误差包含夹爪和抓放阶段，不能单独代表本轮接近能力。

对可疑结果用相同参数、seed=1001 和 1002 重复换图评估，另存输出。若只有浮点级变化，不算有用的视觉响应；不同训练设置应比较相对尺度和跨 seed 一致性。

为避免总体均值掩盖左侧失败，另将 `--episodes` 依次换成第 4 节五组验证编号，分别保存离线 JSON。不要使用脚本默认的 63–89，因为其中包含本轮训练集。

候选模型选定时记录实际 checkpoint 步数和目录，不能只记会变化的 last。

## 8. 检测定位的检查与现有工具边界

第 5 节叠的是人工真值框，不能当成模型预测框。当前两份离线评估脚本没有给出完整的预测框可视化、IoU/召回率或静止杯心抖动报告。

因此，如果要独立验收“检测器在五方位是否定位正确”，还需要补充并验证 FCOS 预测解码/叠图评估工具，或使用项目中已验证的等价工具；本指南不虚构一个现成命令。预测框需与 XML 在同一图像坐标系下匹配，回归解码需核对 stride。

这不阻塞前面的 ACT/ACTDet 训练和动作诊断，但在没有预测框证据时，不应把真机失败断言为检测失败或宣布检测达到某个召回率。

后续补齐检测报告时，先固定阈值，在留出 18 集按方位统计可见杯子的召回、IoU、误检；遮挡/不可见单列。90 个首帧可用于全数据质量检查，但包含训练样本，不能作为泛化成绩。静止抖动从真实静止片段计算，并注明分辨率；5–8 px 可作为初始参考，需结合抓取所需精度调整。

## 9. 真机测试前的状态和相机核对

在连接机器人电脑上准备相同代码、数据集和候选 checkpoint，并设置 PROJECT、DATA_ROOT。服务器通常没有机械臂，不在服务器上执行控制命令。

重点核对：

- top 是桌面全局视角，gripper 是腕部视角；USB 编号不等于相机身份，每次重连后查看 `outputs/camera_check/`。
- 分辨率、相机固定位置、杯子、背景、照明尽量匹配采集条件。
- 9 维 state 的含义和顺序必须一致，不能只检查维数。当前控制脚本说明 master_gripper.pos 在无 leader 时为常数 0；如果训练演示的该维在接近阶段不是相近常数，就存在输入分布差异，需要先检查并处理，不能把结果直接归因于视觉。
- robot.id 使用已有校准对应的实际名称，port 使用当前设备；示例 nn 和 /dev/ttyACM0 不保证适用于每台机器。

本脚本会发送夹爪动作，不会自动检测杯子接触或冻结夹爪，也不会自动把机械臂返回初始姿态。需要现场操作者使用已验证的复位方式；不要在电机保持目标时强掰关节。

## 10. 五方位真机接近测试

每个候选模型左、左前、正前、右前、右各 5 次，共 25 次。先用训练覆盖范围内的五个桌面标记，按“左→右→正前→左前→右前”循环五轮，避免连续只测同一侧。两模型使用相同杯位与初始姿态。

预先确定“接近目标”：沿演示接近方向、接触杯子前的末端位置。建议先设安全间隙 3–5 cm，具体根据杯子和夹爪几何现场固定。它是人工观察停止线，不是程序里的自动距离阈值。

以下示例每次只执行一轮，便于在下一轮开始前完成复位。先替换 CKPT、robot.id、port 和相机编号，再执行：

```bash
export CKPT="$PROJECT/outputs/train/C_DET_s1000/checkpoints/last/pretrained_model"
export TRIAL=DET_left_01
mkdir -p outputs/formal1_C_phase1/real_robot
python -m lerobot.scripts.control_act_det \
  --policy.path="$CKPT" \
  --policy.n_action_steps=1 \
  --policy.temporal_ensemble_coeff=0.01 \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 --robot.id=nn \
  --robot.max_relative_target=5.0 \
  --robot.cameras='{top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, gripper: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}' \
  --dataset.repo_id=QYyyyyyyy/formal1_C \
  --dataset.root="$DATA_ROOT" \
  --dataset.single_task="Pick up the half-filled transparent plastic cup steadily without spilling" \
  --dataset.num_episodes=1 --dataset.episode_time_s=15 \
  --dataset.reset_time_s=15 --dataset.fps=30 \
  2>&1 | tee "outputs/formal1_C_phase1/real_robot/$TRIAL.log"
```

`max_relative_target=5.0` 沿用之前尝试过的限制，它不是 5 cm，也不是硬件限速/防碰撞保证。两模型采用同一控制设置，记录实际循环频率；持续达不到 30 Hz 时先处理延迟或明确降低频率的实验条件。

必须在有本地键盘监听的交互式桌面运行。脚本在 headless 模式会跳过人工相机确认，不能依赖远程无界面的 Esc/方向键停止。先在可控条件下确认停止操作和实体急停可用。

按右方向键提前结束当前轮，Esc 结束测试；到达停止线或运动明显错误时提前停止，15 秒只是时长上限。录像由外部相机或现有录屏工具保存，控制脚本不会自动录视频。

每轮记录以下字段，可建表手动填写：

```text
日期,代码commit,checkpoint步数,模型,方位,轮次,杯位标记,
初始姿态一致,相机核对通过,初始方向正确,到达接近区,
横向偏差cm,测量方式,接触杯子,持续抖动,实际控制Hz,
人工停止原因,录像路径,日志路径,备注
```

成功建议定义：初始运动方向正确，持续接近对应杯位，到达预先划定的接近区域，无接触、明显振荡或越界。横向偏差可先采用 ≤3 cm；使用尺子/桌面网格和固定拍摄视角估计，未经标定不能把像素直接当厘米。每轮使用相同定义。

首轮通过目标：至少 23/25 成功，且每个方位至少 4/5。该小样本只是阶段验收，不是对真实成功率的置信保证。通过后换五个相邻但不同的杯位重复测试，防止只验证少数固定点。

## 11. 如何决定下一步改动

| 结果 | 下一步 |
|---|---|
| 标注错帧/相机反置/输入状态不一致 | 先修数据或输入链路，重新固定版本再训练 |
| 两模型训练损失下降，但验证动作差、换图响应弱 | 检查分方位及接近阶段表现，再做同初始姿态、跨杯位对照；不要仅靠增加训练步数 |
| 检测框确实准确，但策略不随杯位改变 | 优先研究视觉到动作的注入与状态捷径；单独比较 fcos_feature_inject 等改动 |
| 检测框漏检或定位差 | 先查漏标/负样本和遮挡，再尝试受控增强或检测预训练；每次只变一类因素 |
| 离线较好，真机差 | 优先核对相机视角、处理器、9 维状态、校准、控制延迟及开环动作执行长度 |
| ACT 和 ACTDet 都通过 | 记录两者成本和稳定性，保留更简单的可用基线；不能预设检测分支必然更优 |
| ACTDet 稳定通过且优于 ACT | 固定模型和控制设置，换 seed/相邻杯位复验，然后进入抓取、拿起、放下时序阶段 |

暂不同时改 KL 权重、网络深度、动作块长度、检测权重、Mask 和控制参数。阶段二需要另设抓取闭合、抬升保持、放下释放三个判据，不能用本轮“接近成功”替代完整抓取成功。

## 12. 交付清单与当前最先做的事

- [ ] 记录服务器代码 commit、数据 revision、环境版本。
- [ ] 生成并人工查看标注检查图片，解决可见杯子漏标和错位。
- [ ] 保存 split.json，确认训练命令只含 72 集。
- [ ] 完成 ACT/ACTDet 各 2k 冒烟和 20k 首轮训练。
- [ ] 保存零 z、视觉敏感性、分方位离线报告，明确现有指标边界。
- [ ] 核对真实 9 维状态与相机身份；准备实体机器人和录像。
- [ ] 完成候选模型五方位接近测试并记录实际停止原因。
- [ ] 根据结果更新工作日志，决定继续训练、修感知或改视觉动作连接。

现在从第 3–5 节开始：先下载并固定版本，再做标注叠图验收。图像、轨迹和标签对应正确后，才值得开始两组训练。
