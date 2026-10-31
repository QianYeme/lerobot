# Post-selection qualitative evidence — not construction supervision

图：figures/auto_selected_temporal_tracks.png。四行是算法已选出的ep1171/1187/1151/850；红框是光流跟踪的自动patch，不是人工GT框，也不是杯/汤罐标签。所有原图来自Train35。

执行者在算法和gate输出完成后查看了这张图：ep1171与ep1151选区看起来在桌面/机械臂边缘，ep850在机械臂部件附近，ep1187更接近被操作物体区域。这是定性观察，不是4个prototype的标注准确率。

可见的失效模式是：本来静止的背景在机械臂移入/遮挡后，光流会被运动前景带走；机械臂某一局部本来也可能在闭爪前静止、之后运动。两者都会满足“先静后动且邻近末端”的proxy。DINO的上下文特征一致性和单一速度相关性不能保证该patch确实是被抓目标。

因此即使候选计数门槛放宽，仍不能仅凭通过proxy差值宣称找到了alphabet soup。没有用本次人眼查看去重新挑选四个中心，也没有改当前算法或阈值。
