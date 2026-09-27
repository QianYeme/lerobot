# formal3 动作区间审核规范

旧版“四个单帧”规范已停用。闭爪和释放都是持续动作，不能要求人工从连续运动中选出唯一正确帧。

## 三个审核项

1. **闭爪区间**：夹爪开始持续向内运动 → 闭合动作基本稳定。
2. **抬升开始**：机械臂真正开始向上或搬运杯子的边界；允许晚于闭爪稳定。
3. **释放区间**：放置后夹爪开始持续张开 → 张开动作基本稳定。

系统依据每集 `observation.state.gripper.pos` 相对行程的 10%–90%给出闭爪/释放建议区间；视频用于审核语义。`load` 只作诊断，不决定边界。

## 使用网页审核台

```powershell
python scripts/run_phase_review_ui.py
```

浏览器打开 `http://127.0.0.1:8765`。区间起止和抬升边界均可手填或读取当前视频帧；确认所有边界后点击“确认通过并进入下一待审”，整个 episode 记为人工审核通过。修改已确认 episode 的任何边界或备注都会自动撤销确认。

新版结果保存到：

- `../review_ui_v3_reviewed/phase_step_reviews.json`
- `../review_ui_v3_reviewed/phase_step_reviews.csv`

旧版 `review_ui/`、`review_ui_v2_intervals/` 和四联图只保留审计，不得与新版结果混用。
