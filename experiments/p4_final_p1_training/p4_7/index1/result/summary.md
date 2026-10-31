# P4.7 私有模型发布与现场交接 result

RELEASE_COMPLETE=true；RESULT_READY=true（发布/指南/离线验收）；P4_7_COMPLETE=false（现场尚未执行）。

七个独立私有repo均实际创建、提交，远端大小/LFS SHA通过，新目录按revision下载后逐文件SHA通过。真实Dev15流式source/export输出差0、reset差0，Train90归一化通过；Diffusion独立下载复验覆盖9帧、跨原生8-action重规划边界，NewGL Local safe/source参数一致、ON/OFF arm不变。只读补充诊断和指南最终revision再按全部文件SHA确认。source比较复用同一runtime，不冒充独立算法复现。

## 实际发布清单

|角色|私有仓库|最终revision|
|---|---|---|
|L1|[QYyyyyyyy/lerobot-p4-l1-legacy-p1](https://huggingface.co/QYyyyyyyy/lerobot-p4-l1-legacy-p1)|fa5eb1d861c649f296e854da2fb16642f434329f|
|F3|[QYyyyyyyy/lerobot-p4-f3-fixed-p1](https://huggingface.co/QYyyyyyyy/lerobot-p4-f3-fixed-p1)|d768d3e0d0ae497e758fe11555d66ead34d57526|
|A1|[QYyyyyyyy/lerobot-p4-a1-act-matched](https://huggingface.co/QYyyyyyyy/lerobot-p4-a1-act-matched)|d1f89fc240fa6436d7a5ea4a0ddca1ece763f6c1|
|VACT-S1|[QYyyyyyyy/lerobot-p45-vanilla-act-s1](https://huggingface.co/QYyyyyyyy/lerobot-p45-vanilla-act-s1)|8203d2acfb03edfc1a76afce6990cb0942d14649|
|VDP-S1|[QYyyyyyyy/lerobot-p45-vanilla-diffusion-s1](https://huggingface.co/QYyyyyyyy/lerobot-p45-vanilla-diffusion-s1)|3f9102dd1c58ec6004a07b44784e730bf9b1716e|
|NG-L-GL|[QYyyyyyyy/lerobot-p45-newgl-legacy-gl](https://huggingface.co/QYyyyyyyy/lerobot-p45-newgl-legacy-gl)|a7c0deeaca4a700023f75db3317cd230a758bd3b|
|NG-F-GL|[QYyyyyyyy/lerobot-p45-newgl-fixed-gl](https://huggingface.co/QYyyyyyyy/lerobot-p45-newgl-fixed-gl)|dcf3af3239a0c1fa65bcab4d3d2eaa7c42bad286|

## 现场步骤与边界

七条独立直接运行指令：../REAL_ROBOT_TEST_GUIDE.md；完整324计划/核心前导60：../trial_plan.json。
使用tools/p47_download.py和result/upload/upload_receipts.json下载到控制主机并验SHA。控制主机本地模型加载、相机、标定和实际smoke均PENDING；未接机器人、无Final15、未填成功率。无杯只Shadow，补充工具记录真实object tokenizer中心/相似度/patch匹配，不发送动作。

ACT部署1/.01为显式覆盖，Diffusion原生2obs/16horizon/8execution/100DDPM。NewGL Legacy绑定60k、Fixed绑定100k、各10k Local；不能换Global。L1/F3不同seed、NG所选步数不同，不能做严格单变量固定步数因果归因。旧P4.5错误指标作废。

用户明确授权凭据使用覆盖原要求，凭据不进代码/日志/ZIP；所有新repo保持private。发布遇到分片/元数据网络问题，最终使用服务器network_turbo、TLS1.2、Xet及按SHA服务器端复制与幂等重试完成；以实际receipt为准。

## 清理

按用户授权删除六个一次性smoke checkpoint约6.25GiB，保留数值/失败日志与全部正式checkpoint；另清pip缓存387文件约2.9GB（系统盘）。删除前SHA与路径见cleanup/cleanup_manifest.json。本轮发布不需要重训或清理正式候选。后续可清理HF已验证后的服务器重复导出权重，需保留receipt/manifest。

## 后续

发布交付完成后进入用户明确授权的P4.8独立仿真G0–G1；不等待现场真机完成，不以仿真替代真机证据。P4.8的G2原型语义若无训练集可追溯目标区域，必须STOP+PROPOSAL，不能擅自聚类后训练。独立Builder/Red-Team审核PENDING；本包自查不替代审核。


## 已完成的额外去重清理

HF最终私有revision和独立下载全文件SHA验证通过后，已删除staging中约3.13GiB重复部署权重/backbone副本。完整可运行副本仍在download_verify，所有正式训练源checkpoint保留。连同6.25GiB一次性smoke模型，数据盘共释放约9.38GiB；另清理约2.9GB系统盘pip缓存。删除前哈希、剩余副本路径与HF revision见cleanup/verified_staging_duplicates.json。staging的权重已刻意清除，未来重发布需先重新导出。
