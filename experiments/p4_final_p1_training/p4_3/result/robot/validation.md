# 验证

- 导出权重逐 tensor 精确回读：四模型 PASS。
- F3 原始 checkpoint hash：与 decision 一致。
- state8D/action6D、正式步数：四模型 PASS。
- 固定 Dev15 每模型 90 个采样点：部署入口与原 checkpoint 参考预测最大差异 0。
- reset 重复输出差异：四模型 0。
- 160k 零图像 synthetic runtime dry-run：PASS；不是实际定位验证。
- 本地新增 Python 文件 py_compile：PASS。
- control_f5.py git diff --check：PASS。
- nn 标定、摄像头编号/RGB方向、物理控制时延、真机动作链：PENDING。
- 无新训练、无 Final15 模型评估、无机械臂操作。

- 48 个计划槽位、每模型 12 次、位置/重复无缺项：PASS。
- CLI --help 包含四模型：PASS。
- 新增源码 trailing whitespace/secret 检查：PASS。
- HF 最终 revision 的交付文件大小和 LFS SHA：PASS。
- 发布清单已剔除运行产生的 __pycache__/pyc；未重新上传或修改权重。
