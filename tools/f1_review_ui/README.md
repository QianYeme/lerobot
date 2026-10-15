# F1 独立 RGB 盲审页面

在项目根目录启动（本机 Python 需有 OpenCV，ffmpeg 在 PATH）：

```powershell
python scripts/run_f1_review_ui.py
```

打开 http://127.0.0.1:8766 。首次启动自动把六段 mp4v 视频转成浏览器可播放的 H.264，缓存写入 `outputs/f1_rgb_review_ui/video_cache`，不修改原始证据。端口被占用时可加 `--port 8767`，不要结束不明服务。

## 操作

1. 点击 Train 0，填写审核者；按顺序审 Train 0/2/4/6/7/9。
2. 先连续播放完整双视角视频，再在接触附近暂停。左右箭头逐帧、Shift+箭头十帧；可拖动进度条、输入帧号跳转、选择0.25倍速。暂停后显示原始证据视频中该帧的直接解码图。
3. 点击“添加一段区间”，选择标签。可靠区间填可靠开始/结束，按钮“取当前帧”把当前显示原帧号填入该格。两端包含；填写看到的证据和判断理由。
4. 只标确定的内部帧，不要求猜准接触的第一瞬间。不同标签可有空隙，未知边界不硬填。为不确定部分单独添加 UNKNOWN/OCCLUDED，填“不确定开始/结束”和原因，可靠帧留空。所有帧均指原episode帧号。
5. 点击“保存到 CSV”，等待“已写入并回读确认”。每次保存只替换当前集的行，其余集保持，CSV整体原子替换；旧文件自动备份在同目录 `csv_backups/`。
6. 刷新页面会读取磁盘结果。未保存修改有离开提示，并保留本浏览器草稿；出错时不要刷新，先“下载草稿 JSON”。若保存提示冲突，是CSV已被另一页面/程序修改，先导出草稿再重新载入，不能强行覆盖。

直接写入：

```text
experiments/force/f1_formal4_signal_feasibility/index1/result/evidence_samples/review_intervals.csv
```

页面不提供信号统计、原始特征、任意文件或Dev访问。接触与阶段是不同轴，PRECONTACT可与CLOSING重叠；同轴不同可靠类不可重叠。`evidence`列由服务自动填写视频名与帧区间。保存只是保存已审区间，不代表看完整集、不自动标整集通过、不自动选择任务或运行probe。

当前合同先看前六集，接触主标签优先PRECONTACT/CONTACT_SUPPORTED；阶段标签释义在页面中。六集都保存后告知执行端，再按合同计算覆盖门（主窗口还需合法历史），不要自行打开Dev或根据预测精度选任务。

## 验证

```powershell
python -m pytest tests/scripts/test_f1_review_ui.py --confcutdir=tests/scripts -q -o addopts= -p no:cacheprovider
node --check tools/f1_review_ui/app.js
```

端到端浏览器 smoke：`python scripts/smoke_f1_review_browser.py`，需已生成视频缓存、系统Edge以及安装在 `outputs/f1_rgb_review_ui/test_dependencies` 的Playwright。它使用临时CSV，不向正式文件写任何测试标签。此依赖仅用于测试，不是页面运行依赖。
