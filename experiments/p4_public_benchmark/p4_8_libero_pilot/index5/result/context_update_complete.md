## 2026-10-31 P4.8-r5服务器离线pilot全部完成：三模型均4/10，对象路径作用极弱

- 三路各正式10000步完成，完整5000/10000 checkpoint及optimizer/RNG保留。全部Dev9 1391帧/模型、27逐episode记录、30次配对仿真、54条对象ON/OFF诊断及孤立延迟/参数量已完成，无本地推理/真机。G2/G3/G4工程交付通过，不代表性能领先。
- ACT、Legacy、Fixed均4/10；成功init完全相同10–13，14–19均280step内未成功；三模型10组起始公开8D状态逐元素相同（最大差0）。1task/seed1/10k pilot，不是10任务正式benchmark，不能据此选定优胜模型。
- 末期gate Legacy=-6.75154e-6、Fixed=-4.14562e-7；各27个Dev锚点h0 arm raw ON/OFF均差7.46486e-8/1.90307e-8、gripper2.38706e-7/8.36021e-8。策略在受检锚点上对对象残差几乎不敏感，不证明全训练过程/所有状态没用过该路径。
- 后续只读CPU检查：gate在AdamW组内、requires_grad=True、step10000且矩非零，排除简单optimizer漏登记；没有追加训练更新。下一轮建议围绕对象路径使用弱进行诊断，不擅改gate/init/LR或扩大任务。
- 三路实测总速度约为串行2.09–2.37倍，数值更新可重复；正式训练墙钟ACT约14.93min、两P1约17.1min；控制器含吞吐/训练/评估/自动交付约22min。推理batch1整chunk100约8.56/17.02/17.25ms，不当作单动作周期。
- 原始自动ZIP 35,786,315bytes，SHA6b89bab458a12d1fd69f356564b48e78dc74892e15de22beda15a868770702e2；本地CRC/全部打包SHA通过，服务器原manifest256项（含大模型/缓存指针）全部SHA通过。追加审阅版含关键机制发现、配对验证和DINO源码快照；原自动包不改。
- 服务器队列已退出，无活动训练/评估，GPU2MiB/0%，磁盘约18.44GiB空闲。checkpoint和原始日志/视频不删。独立Builder/Red-Team PENDING，无Final15/真机/新seed/task/第四模型/HF/Git发布。
- [r5执行后核对](../experiments/p4_public_benchmark/p4_8_libero_pilot/index5/result/executor_postrun_review.md)；待用户与规划助手审核result/proposal.md后再进入下一科研轮次。
