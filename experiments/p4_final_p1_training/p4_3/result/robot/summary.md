# P4.3 四候选部署准备

四候选：L1 80k、F3 80k、A1 80k、用户追加的 L1_160K 160k。
所有权重从实际 resume.pt lossless 导出 safetensors，逐 tensor roundtrip bitwise equal。
F3 原始 resume SHA 与 decision 一致；各 checkpoint step、8D state/6D action 均核对。
部署 wrapper 在冻结 Dev15 每模型 90 个进度采样点与参考代码逐预测比较：最大误差均为 0；reset 前后动作差异均为 0。
F3 参考来自隔离 fixed ACT 源码，其余来自 Legacy/Original ACT 源码与原始 checkpoint。
该离线检查使用真实观测与示范动作，属于 teacher-forced evidence，不代表闭环真机成功。
仅评估固定 Dev15，未读取 Final15。本轮未连接或操作机械臂，硬件预检与实际行为结果待现场完成。
服务器 RTX5090 的 chunk wrapper 均值约 ACT 13.36ms、P1 18.44–18.65ms，不包含相机、串口、完整控制循环；不保证现场达到 30fps。
原三模型 36 次试次与 160k 的独立 12 次补充共 48 次；尚无真实试次结果，不生成伪成功率。

模型仓库：https://huggingface.co/QYyyyyyyy/P4_3_FORMAL6_SO101
最终 commit：a0e55ec840709cf0cacb59ccb26a3e36dffd11ec
远端 226 个交付文件大小及 LFS SHA 已核验。ZIP 只含代码、指南与证据，不含大权重；权重从 HF 获取。
