# F1 独立 RGB 盲审：只审核前六个 Train

本目录仅包含RGB、帧索引和空白审核表，没有load/current曲线或probe分数。审核者不要打开上级signal_integrity.json、figures或raw特征。历史D2.1事件不直接转作F1标签。

按冻结manifest顺序播放以下完整双视角片段，然后在关键事件附近逐帧核查；左top、右腕部，30FPS。片段从原episode第0帧开始，原帧号=播放器时间×30。JPG文件名为原帧号，仅作定位，不替代连续观看。

1. [Train 0](train_ep000_dual_view.mp4)
2. [Train 2](train_ep002_dual_view.mp4)
3. [Train 4](train_ep004_dual_view.mp4)
4. [Train 6](train_ep006_dual_view.mp4)
5. [Train 7](train_ep007_dual_view.mp4)
6. [Train 9](train_ep009_dual_view.mp4)

填写review_intervals.csv：一行一个可靠区间，可添加行，记录label、first_reliable_frame、last_reliable_frame、边界不确定区间、审核者及依据。不确定区间不要混入可靠区间。尚未审核不要填UNKNOWN，UNKNOWN是已审后无法判定。

- PRECONTACT：可见夹爪与杯子没有接触，接触前闭爪也可以属于此类。
- CONTACT_SUPPORTED：可见接触，或后续可见杯子与夹爪共同运动支持此前接触；不要求证明杯子离桌。不以单纯闭爪、load/current或静止姿态判定。
- CLOSING：闭爪动作过程，不等于接触。
- POST_CLOSE：闭爪结束至释放前，不等于抓稳或顶部保持。
- RELEASE：张爪释放过程，不等于杯子已确认稳定落桌。
- UNKNOWN/OCCLUDED：给出遮挡、透明边界或不同步等具体原因。

第15帧历史窗口还要求t−2等合法历史，目标帧必须≥16。每类至少5个可靠目标帧，区间可不连续但必须逐段有证据。

合同分支：前六集至少4集同时有PRECONTACT和CONTACT_SUPPORTED才继续全部Train接触审核；否则转全部Train三阶段审核。随后满足24/36覆盖才冻结Train任务；冻结并哈希后才审核Dev。不得根据probe成绩选任务。

当前状态：视频已准备并核对帧数，尚未完成本轮独立连续播放审核；不是“六集均不可判”。
