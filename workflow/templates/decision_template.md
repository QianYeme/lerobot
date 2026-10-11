# Experiment Decision Contract

> 本文件由用户与 ChatGPT 共同冻结。Codex 只读。
>
> Revision: r0

---

## 1. Identity & Mode

- Experiment ID:
- Track:
- Stage:
- Index:
- Mode: `ENGINEERING / DIAGNOSIS / DISCOVERY / CONFIRMATION`
- Dataset:
- Code State:
- Previous Experiment:

### Current Authorization

本轮只授权：

> ...

本轮明确不授权：

- ...

---

## 2. Research Question

> **RQ:** ...

只允许一个 Primary Research Question。

---

## 3. Paper Relevance

### Candidate Claim

> ...

### Why It Matters

> ...

### Non-Claims

本轮即使 PASS 也不能证明：

- ...
- ...

---

## 4. Baseline / Intervention

### Baseline

...

### Intervention

...

### Only Intended Scientific Difference

> ...

---

## 5. Frozen Contract

### Data
...

### Observation
...

### Action
...

### Model / Controller
...

### Training
...

### Evaluation
...

### Deployment
...

### Allowed Engineering Degrees of Freedom

Codex 可以自主：

- ...

---

## 6. Assumptions / Open Questions

- `[FROZEN]` ...
- `[PROVISIONAL]` ...
- `[OPEN]` ...
- `[REJECTED]` ...

---

## 7. Hypothesis & Falsification

### H1

> ...

### Expected Mechanism

1. ...
2. ...

### Reject H1 if

- F1:
- F2:
- F3:

### Forbidden Post-hoc Rescue

本轮结果产生后不得：

- 改 PASS threshold；
- 换 primary metric；
- 删除失败 episode；
- 换 Final Test；
- 临时加模块；
- 只报告最好 seed/checkpoint。

---

## 8. Gates & Metrics

### Mandatory Gates

- G...
- G...

### Primary Metric

> ...

### Secondary Metrics

- ...

### PASS

> ...

### FAIL

> ...

### STOP / BLOCKED

> ...

---

## 9. Authorization

### Codex MAY

- ...

### Codex MUST

- ...

### Codex MUST NOT

- 修改本 `decision.md`
- 改 baseline / intervention / split / primary metric / PASS threshold
- 自行进入下一科研阶段
- ...

### Requires User Approval

- 新研究结构
- Discovery → Confirmation
- Full Training
- Formal Final Test
- Formal Real-Robot Test
- Push / Publish
- ...

---

# Conditional Sections

仅在适用时保留。

---

## A. Confirmation Section

- Seeds:
- Checkpoint Selection Rule:
- Final Test:
- Statistical Comparison:
- Confirmation Invalid If:

---

## B. Real-Robot Section

- Robot:
- Start State:
- Placement Region:
- Trial Count:
- Stage Definitions:
- Full Success:
- Safety Limits:
- Stop Conditions:
- Video / CSV Evidence:

---

## C. Data-Collection Section

- Scene:
- Nominal Region:
- Sampling Plan:
- Episode Count:
- Task Timing:
- Cup / Liquid:
- Operator Instructions:
- QC:
- Split Rule:

---

## D. Model / Architecture Section

- Inputs:
- Fusion Point:
- New Parameters:
- Gradient Path:
- Detach Rules:
- Initialization:
- Backward Compatibility:
- Default-Off Behavior:

---

## E. Force / Control Section

- Signals:
- Contact Definition:
- Grasp Definition:
- Hold Definition:
- Release Definition:
- Controller Ownership:
- Fallback:
- Threshold Source:
- Calibration Evidence:

---

## Required Deliverables

Codex 必须写入：

```text
result/
├── summary.md
├── metrics.json
├── validation.json
├── figures/       # 若适用
└── proposal.md    # 若有新建议
```

Raw evidence：

```text
results/experiments/<experiment_id>/
```

完成本轮后停止，不得自行生成下一轮 decision。
