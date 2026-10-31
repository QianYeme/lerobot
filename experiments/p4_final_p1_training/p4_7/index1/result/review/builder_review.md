# Builder 执行方自查（独立审核PENDING）

- 源身份：P4.3 manifest确证L1/F3/A1，三个source SHA匹配；A1明确ACT-MATCHED-8D，非stock。
- 新模型：VACT-S1 80k、VDP-S1 100k；NG Legacy Global60k+Local10k，Fixed Global100k+Local10k，Local绑定原Global SHA。
- 安全权重导出逐tensor一致；真实Dev15样本流、reset和归一化通过。独立HF下载时覆盖Diffusion8-action重规划边界（9帧），Local参数与源逐tensor检查。
- 部署ACT1/.01为显式覆盖，Diffusion原生8执行不套ACT参数。统计量/原型/离线backbone与冻结source/src随包，不把source路径猜成权重。
- 七命令dry-run无硬件；无杯Shadow分支/补充诊断脚本无send_action；控制主机和现场安全仍PENDING，不填成功率。
- 最终HF发布状态必须逐一以receipt的PRIVATE和revision/下载SHA为准；上传过程重试不能算完成。
