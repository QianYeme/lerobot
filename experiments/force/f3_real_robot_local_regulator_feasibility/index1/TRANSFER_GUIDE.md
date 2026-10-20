# F3 模型迁移包（本轮不执行真机）

Hugging Face：`QYyyyyyyy/F3_C1_LOAD_SO101`（默认私有；使用本人账户权限下载）。

包含F2.3确切C1-load组合：A0 retry1 80k全局6D OC-ACT、C1-load10k
局部MLP、Train归一化、杯子原型、DINO/ResNet构造缓存、训练源码快照和校验器。
不需要下载完整训练数据集；16条Dev数值校验输入只是离线测试夹具，不是真机记录。
不含Final12、设备标定文件或任何token/SSH密码。

## 新系统先做离线检查

建议Python3.12；本包服务器验证环境Python3.12.3、torch2.8.0+cu128。
按新系统CUDA/驱动安装兼容torch/torchvision，再安装源码依赖。

```bash
# 先通过 hf auth login 或环境变量完成本人认证，不要把token写进脚本/仓库。
hf download QYyyyyyyy/F3_C1_LOAD_SO101 --local-dir F3_C1_LOAD_SO101
cd F3_C1_LOAD_SO101
export TORCH_HOME="$PWD/torch_cache"
export PYTHONPATH="$PWD/source/src"
python source/scripts/verify_f3_bundle.py --bundle "$PWD" --device cuda
```

没有GPU时可用`--device cpu`，速度较慢。源码快照不等于完整Python环境：
先参考`environment_versions.json`与`source/pyproject.toml`安装依赖。
不要无差别照搬服务器CUDA wheel到不兼容系统。

## 重要：不是现成真机启动指令

现有`control_act_det.py`没有局部MLP开关，不能直接宣称支持regulated推理。
随包`local_load_regulator.py`是独立、无robot I/O的适配器，不会打开串口。
后续在新系统接入控制循环时必须：

1. 全局输入仅六个位置，顺序：shoulder_pan、shoulder_lift、elbow_flex、
   wrist_flex、wrist_roll、gripper；全部为`.pos`。RGB语义不变。
2. 全局观察/动作按`normalization.json`的Train均值/标准差变换，RGB按
   ImageNet均值/标准差；本checkpoint没有标准policy_preprocessor.json，
   不能假定通用record脚本会自动完成相同处理。原型绝对路径由校验脚本重定位。
3. 局部输入是同一步的原始load、原始gripper.pos、已反归一化的nominal
   gripper位置；适配器内部做一次归一化，不能把load预先归一化后重复传入。
4. 局部修正只加在夹爪列。全局后处理/既有temporal ensembling之后、
   原有机器人安全限幅之前接入；不得改变机械臂列或关闭既有限幅。
5. `disabled`返回原动作且不运行局部MLP；`shadow`只算并记录假设动作，
   实际动作不变；`regulated`才返回修正动作。默认必须disabled。
6. 保留F2.3上限±4.832699459791184位置单位，不是牛顿。不得静默放大。
7. 既有机器人标定/相机编号/方向/控制频率/限幅/平滑需在新系统现场核验。
   checkpoint配置记录n_action_steps=1、temporal_ensemble_coeff=null、chunk100；
   这不是替现场冻结部署配置，也不是授权随意变更实机参数。

## 后续真机安全门（尚未执行）

按F3 decision先完成部署硬件Stage0与每步遥测/视频，再做3次shadow。
shadow审核通过才允许同全局模型baseline3次、regulated3次。
未满足条件不得跳到regulated。所有中止/危险事件保留，出问题停止，不调参续跑。
不读Final12、不新增训练或高频控制。

离线F2.3只支持局部夹爪误差改善；C1继承A0的臂误差，比C0高约17.48%。
不保证闭环稳定、抓取成功、接触因果或物理力控效果。F3_COMPLETE仍为false。
