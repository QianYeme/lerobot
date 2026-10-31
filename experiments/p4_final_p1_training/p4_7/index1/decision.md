# Decision P4.7-r0 — Hugging Face 模型归档与 SO-101 真机对照

状态：FROZEN FOR CODEX EXECUTION（尚未执行）  
日期：2026-10-30  
工作流：用户/ChatGPT 冻结决策；唯一的本地 Codex 通过 SSH 操作远程服务器，回本地执行部署。Codex 不得改变科学问题或绕过验收。

## 0. 本次要完成的事

1. 从 P4.5 最终结果清单核对所有正式完成的模型及其精确 checkpoint、SHA256、代码血缘与推理依赖。
2. 将需要真机测试的模型作为**各自独立的 Hugging Face model repo**上传到 `QYyyyyyyy` 命名空间。默认 PRIVATE，确认访问权限之后再考虑公开。**不上传原始 resume.pt 中的 optimizer、训练数据、令牌等非部署内容**；优先发布与架构匹配的 `save_pretrained` 可加载部署包，附来源 manifest 与推理说明。
3. 从 Hugging Face 仓库在独立本地目录重新下载、加载、进行输出等价性/预处理预检，再开始真机对照。
4. 实施正常场景、光照/水量鲁棒性、无杯 Shadow 诊断；明确区分常规任务成功率和目标缺失行为。
5. 将最终机器人逐次结果与服务器离线结果分离归档，不碰 Final15。

## 1. 安全与权限

- 对话中已暴露 HF Access Token：立刻在 HF 设置撤销旧令牌并创建新的最小权限写入 token。**禁止使用、记录或传播已暴露 token**。
- 通过操作者手动设置环境变量 `HF_TOKEN` 或本地凭据管理；不可把值写入命令回显、shell history、源码、commit、README、日志、trace、截图。调试仅报告 token 是否存在，不能打印其内容。
- 仓库默认 `private=True`；上传需当前 Codex 对 HF 认证有明确权限，无法完成时提交 BLOCKED_UPLOAD 的证据，不得声称上传成功。
- 操作者现场监督真机、校准和急停；无杯条件只做无动作 Shadow，不给机器人执行危险预测。

## 2. P4.5 证据与模型边界

数据来源：`P4_5_UNBOUNDED_FINAL_result.zip`，内嵌 `P4_5_server_result.zip`。

- VACT-S1/S2/S3 各 80,000 更新，官方 stock LeRobot v0.4.4 commit `8fff0fde7c79f23a93d845d1a50e985de01f8b8a`，6D、两相机，`batch=8`。与旧 A1 (ACT-MATCHED-8D, batch4, gripper x3, backbone LR 1e-4) 绝不可混淆。
- NG-L-G / NG-F-G 各 100,000 更新；NG-L-L / NG-F-L 各 10,000 更新，Global 冻结。**NG-L-L 使用 Legacy Global 60k；NG-F-L 使用 Fixed Global 100k**，不允许自行换成 Global 最后一步。
- Vanilla Diffusion VDP-S1 已 100,000 更新，默认 6D、两相机、2-frame observation/16-step prediction/8 executed steps；独立正式完成文件存在。Seed2/3 未完成。
- Legacy P1 L1、Fixed P1 F3、项目 ACT A1 为历史 P4.3/P4.6 真机候选；必须在部署 manifest 找出真正的 checkpoint SHA256/位置。**L1/F3/A1 别名必须先确证，不得猜测。**
- `P4_5_UNBOUNDED_FINAL_result.zip` 不含可部署权重，只有服务器清单与路径。任何外部 upload/download 都需要实际执行并验证。
- P4.5 的 `legacy_checkpoint_comparison.csv` 与 P4.1/P4.2.1 旧评估采样配置不同：不得跨表直接将绝对误差当作同一评估协议。优先按模型统一 `h0` 评估并报告环境条件。

## 3. 上传模型与优先级（每个仓库独立）

建议使用下列**新仓库候选名称**，若存在冲突不覆盖；创建前列出已有 repo，提交命名映射并记录实际 repo_id。

| Repo 候选 | 身份 | checkpoint 选择 | 用途 |
|---|---|---|---|
| `QYyyyyyyy/lerobot-p4-l1-legacy-p1` | L1 Legacy P1 | 部署时已测 L1 的精确 checkpoint（可能是 80k，先验） | P1 现场对照 |
| `QYyyyyyyy/lerobot-p4-f3-fixed-p1` | F3 Fixed P1 | 部署时已测 F3 精确 checkpoint（可能 seed3/80k，先验） | 修复版对照 |
| `QYyyyyyyy/lerobot-p4-a1-act-matched` | 项目 ACT1 | 部署时已测 A1 精确 checkpoint | 历史对照，非默认 ACT |
| `QYyyyyyyy/lerobot-p45-vanilla-act-s1` | 官方默认 ACT Seed1 | VACT-S1 80k | 必须上传/试测 |
| `QYyyyyyyy/lerobot-p45-vanilla-diffusion-s1` | 官方默认 Diffusion Seed1 | VDP-S1 100k | 必须上传/试测 |
| `QYyyyyyyy/lerobot-p45-newgl-legacy-gl` | Legacy New-GL | NG-L-G **60k** + NG-L-L 10k | 完整 New-GL 可用性对照 |
| `QYyyyyyyy/lerobot-p45-newgl-fixed-gl` | Fixed New-GL | NG-F-G **100k** + NG-F-L 10k | 修复 New-GL 对照 |

可选仓库：VACT-S2/S3；New-GL Global-only OFF 可通过同一完整模型禁用 Local 在推理时实现（需契约核验），无需重复存放另一份权重；P1 Legacy120k/160k 在可比离线候选确定后再考虑。仓库未创建或权限不足时不得把上表当作已上传链接。

## 4. 每个模型发布包必有文件

- `config.json` 与模型 `pretrained_model` / `model.safetensors`（或实际对应模型支持的安全权重格式），确保可独立实例化。
- `README.md`：架构身份，官方/项目版本、数据版本及 train split，输入字段顺序/单位/摄像头映射，观测历史/动作窗口，归一化和预处理，推理指令/依赖，安全限制，训练步数，实验身份。
- `manifest.json`：repo_id、角色、source_checkpoint_path、source_checkpoint_sha256、权重来源、git commit、训练 seed/step、global_step/local_step、是否 double norm、是否含 load/current、是否官方 stock、验证时间/测试结果。
- 运行所需的统计量、Cup Prototype Bank、DINO 权重/修复版 top_raw 契约、New-GL Local 归一化统计，以及有合法许可的推理代码。大型第三方 backbone 可以按版本固定安全下载源，但离线/本地加载验收须能完整复现。
- 不上传 HF token、校准中的隐私数据、完整 Train/Dev/Final 视频、optimizer state、未经授权的文件。
- 对于不支持安全导出或缺失关键依赖的模型，标记 `BLOCKED_EXPORT` 并提交 PROPOSAL；不要拼凑出表面可加载但推理语义不同的模型。

## 5. 上传过程 / 幂等与断点续传

单个本地 Codex：SSH→服务器检索 manifest 和各个 checkpoint→在服务器（或者可靠传输到本地）组装发布包→`huggingface_hub` API 创建 `private=True` repo→上传→写 repo commit hash / tree→从新目录 `snapshot_download` →逐模型重新加载、同输入输出对照→回传 manifest。大文件建议分 repo 上传，记录完成状态并支持幂等重试；不要重新训练。

使用新令牌环境变量，不允许日志回显。未经用户另行授权不可将模型仓库公开。上传已存在仓库前必须验证身份并防止覆盖。

## 6. 真机实验设计（首轮部署和第二轮正式分离）

### 6.1 分两组推进

**核心组**：L1、F3、A1、VACT-S1、VDP-S1。先通过本地推理/单位/限位/时序/相机校验。标准明亮半杯水每模型 2 次 smoke；失败时停机诊断，不纳入正式统计。

**New-GL 组**：NG-L-GL 与 NG-F-GL，分别对 Global-only (Local OFF) 与 Global+Local (ON) 做相同输入复现，并确认手臂轨迹不因 Local 直接被改写；先各 2 次正常场景 smoke，未通过则停止后续测试。**Local OFF/ON 必须同 Global checkpoint。**

### 6.2 正式有杯试验

建议固定三个源位置（左中右/覆盖黑框区域，实际坐标/相机像素与布置视频记录），目标区固定；2 光照水平 × 2 杯中水量（半杯/空杯）× 3 位置 × 3 重复 = **36 次/模型**。核心五模型共 **180 次**；New-GL 两种 DINO 预处理 × Local OFF/ON 四条件追加 **144 次**（可在核心组结束且部署合格后分批执行），两组总上限 **324 次**。原则上分阶段分日进行；不是要求一次性完成，避免机械臂/操作者疲劳和环境漂移。

为了减小首轮负担，**先进行前导诊断子集**：4 条件×3 位置×1 次×5 核心模型 = 60 次；如果全部接近饱和，再按预先冻结协议继续补全；不可在观察结果后临时只增加某一模型困难条件样本而不记录偏差。

- 固定杯型、摄像头与相机曝光/白平衡策略、机器人 Home、位置坐标、光照档位及测量办法；应记录实际照度或至少固定灯具/相机设置。
- 以完整自主的 Grab→Lift→Hold→Transport→Place→Release（含无洒水、安全）为主终点；各阶段、首次失败阶段、空抓、重复闭合、停滞、目标区误差为次终点。
- 模型交替、在条件和位置内平衡顺序；每次独立重置杯位。重试、人工协助必须单列；人工辅助不计自主成功。
- 记录 per-trial：model_id + repo revision + hash + preprocessing + light + water + position + replicate + result/first failure + video + timestamps + operator intervention。

### 6.3 无杯诊断（不可与成功率混合）

2 光照×3 空场景摆位×3 次×每个部署模型，**只做 Shadow（记录动作但不执行）**。测第一预测块关节/夹爪运动、是否朝向空目标、Object Token confidence/patch matches（有 DINO 者）、模型静止输出。A1 静止不等于目标缺失识别；DINO 空抓趋势也不等于真实抓取失败。

## 7. 统一比较和分析

- 跨模型真实任务采用成功率与明确环境、推理预算及机器人协议比较。Diffusion 的 2 obs/16 prediction/8 step action 与 ACT 不同，必须保留原生配置并报告闭环动作频率/实际时延。
- P1 Old/Fixed 同 seed/数据/训练预算用于 DINO 修复归因；F3 vs L1 若不同 seed，主要是系统级对照。
- New-GL Legacy vs Fixed：尽量匹配 seed/预算/Global 选择规则，但各自选出来的 Global 最优 step 不同（Legacy60k, Fixed100k），故**不是固定训练步数单变量严格对照**；报告这一混杂项。New-GL Local ON vs OFF（同一 Global）才是严格 Local 功能消融。
- 不把 Dev15 教师强制误差视作成功率，不使用 Final15 做选择。

## 8. 单 Codex 调度与完成定义

1. 先静态确认模型清单和准确权重，完成安全新 Token 准备。用户自己撤销旧 Token。
2. SSH 无需另一个 Codex；上传完成后 repo 远程 revision 可查，独立下载可加载、输入输出比较过关。
3. 将本地真机预检与前导 60 次测试作为首个交付；现场由操作者决定放行，Codex 不得自行启动机械臂。
4. 测试结果按既有科研工作流输出 `result/upload/`、`result/robot/`、`result/review/`；提出阻断性偏差时提交 PROPOSAL。
5. Builder / Red-Team Review 必须报告完整数据、失败及局限。

**明确禁止**：再次使用对话中暴露的 HF Token；虚构上传成功；从 zip 冒充权重；把 A1 改名为官方 Vanilla ACT；跨模型错误复用 DINO 预处理；把 New-GL 的 Local 参数搭配错误 Global；无杯无安全约束直接执行动作；修改仍在运行任务；越权公开仓库。
