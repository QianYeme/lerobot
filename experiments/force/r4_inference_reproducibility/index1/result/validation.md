# R4交付核验

## 服务器已执行

- A/B/C层级门、完整D两遍、seed1补证及修复后关键指标已完成；脚本从未修改<0.001阈值。
- 两seed各28,739帧/7,185批，100 horizons×6动作逐元素重复差为0；C固定真实样本的decoded/normalized图像、state/load输入一致。
- independent_cpu_validation.json为本轮另行读取完整chunk/输入manifest、核对h0缓存与5个global误差、身份及12个checkpoint文件的复算。它不重跑GPU，也不冒充独立外部科研审阅。
- local基于冻结同权重和fresh global重组；本轮CPU复核未独立重推所有local。无新视频解码或真机验证。

## 本地交付检查

`delivery_validation.json`记录原ZIP SHA/CRC、manifest成员SHA、模型身份/合同/脚本一致性、正式输出存在性及9模型表交叉检查；交付ZIP另有manifest和receipt。所有PASS须以实际脚本运行成功为准。

本地定向回归使用`--confcutdir=tests/scripts -p no:cacheprovider`隔离根目录硬件fixture，8项CPU门槛/开关控制/既有分析测试通过，不能代替CUDA复现。git diff --check单独执行。

## 尚未完成

- 独立外部Builder Review、Red-Team Review待网页GPT完成，所以R4_COMPLETE=false，虽然技术R4_REPRODUCIBILITY_PASS=true。
- 另一GPU/软件栈、batch1真机ensemble控制链、相机与串口端到端行为未被本次结果证明。
- 最终数据采集、split冻结、规模训练、SWA/SAM及真机均未开始。

原服务器自动报告逐字节保留在server_original/result；原运行中summary/metrics/validation另存progress_archive。交付说明不覆盖原科研证据，也不隐藏historical B失败。
