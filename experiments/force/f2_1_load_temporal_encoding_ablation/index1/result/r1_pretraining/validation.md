# r1 实际准备验证

prepare_f2_1_r1.py实际运行：Train36/Dev12过滤17243/5748帧；6D状态/6D动作、有限性、连续帧验证通过；load标准化仅Train，W15首帧填充且最后一列等于load_t。缓存无current/master，未读Final样本。

tests/scripts/test_f2_1_preparation.py四项PASS：历史因果/填充、episode复位、真正删除反馈列、h0指标只用第0步且逐帧加权。既有G0缓存归约复现PASS，完整模型forward复现及模型smoke尚NOT_RUN。配置规格不是已实现模型；共享权重相等、current在VAE与推理均排除、加载/梯度/显存/延迟均待GPU。

最后已知服务器westb:27435连接被拒绝，未取得GPU状态；未上传或启动任务。原r0验证原件在r0_preflight/。

## r0历史验证（非当前结论）

- 完整阅读本轮r0合同及当前STATUS/科研工作流；没有改合同。
- HF官方API查询到revision后固定读取4份小型JSON，preflight_f2_1.py实际运行成功，fetch_errors为空；JSON原文与SHA256见baseline_metadata/和baseline_identity.json。
- 首次直接urllib取config出现SSL EOF；改用requests有限重试取得成功，没有关闭TLS验证，没有使用来历不明的替代checkpoint。
- 检查Formal4 nomaster元数据8D字段、已发布P1 config/preprocessor及ACT状态投影源码，发现load/current已在既有状态输入内。未证明模型实际使用强度，不把代码可达性当机制有效。
- split只读元数据核对36/12/12，HF训练episode与T0清单相等；未读取任何Final样本、帧或视频。
- 未执行strict weight reload、GPU forward、A1/A2 smoke、训练、Dev性能评估或延迟测试。依合同身份及输入语义门停止，不能标这些项目PASS。
- 新增预检脚本语法与git diff --check的执行结果另行在工作日志记录；输入语义未澄清前不做无效模型开发。

新增脚本 `python -m py_compile scripts/preflight_f2_1.py` 及 `git diff --check` 实际执行通过；不等于模型smoke或身份门通过。
