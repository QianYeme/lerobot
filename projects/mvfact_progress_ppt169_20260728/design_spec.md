# M-VF-ACT Progress Report - Design Spec

> **Refine-Spec mode** — 规范供审核，审核通过后生成SVG。
> 用途：暨南大学本科毕设汇报，向导师李德平老师展示当前工作成果与下一步计划。

## I. Project Information

| Item | Value |
| ---- | ----- |
| **Project Name** | M-VF-ACT 毕设进度汇报 |
| **Canvas Format** | PPT 16:9 (1280x720) |
| **Page Count** | 12 |
| **Design Style** | swiss-minimal |
| **Target Audience** | 本科导师 (李德平) — 了解机器人/深度学习领域 |
| **Use Case** | 毕设阶段性成果汇报 |
| **Created Date** | 2026-07-28 |

## II. Canvas Specification

| Property | Value |
| -------- | ----- |
| **Format** | PPT 16:9 |
| **Dimensions** | 1280 x 720 |
| **viewBox** | `0 0 1280 720` |
| **Margins** | left/right 80px, top 60px, bottom 50px |
| **Content Area** | 1120 x 610 |

## III. Visual Theme

### Theme Style

- **Mode**: `pyramid` — 结论先行：做了什么 → 为什么这样做 → 下一步做什么。适合学术汇报。
- **Visual style**: `swiss-minimal` — 网格对齐，锐利边缘，大量留白，无冗余装饰。适合学术+技术汇报。
- **Theme**: Light theme
- **Tone**: 学术专业、清晰、层次分明

### Color Scheme

| Role | HEX | Purpose |
| ---- | --- | ------- |
| **Background** | `#FFFFFF` | 页面背景 |
| **Secondary bg** | `#F0F4F8` | 卡片背景 |
| **Primary** | `#1A56DB` | 标题装饰、重点色块 |
| **Accent** | `#E0245E` | 数据亮点、创新标注 |
| **Secondary accent** | `#0EA5E9` | 次级强调、图表辅助色 |
| **Body text** | `#1F2937` | 正文 |
| **Secondary text** | `#6B7280` | 注释、图注 |
| **Border/divider** | `#E5E7EB` | 分割线、卡片边框 |
| **Success** | `#10B981` | 已完成标记 |
| **Warning** | `#F59E0B` | 进行中/待完成标记 |

## IV. Typography

| Role | Font Stack | Size |
|------|-----------|------|
| **Title** | "Microsoft YaHei", Arial, sans-serif | 36px |
| **Subtitle** | "Microsoft YaHei", Arial, sans-serif | 24px |
| **Section title** | "Microsoft YaHei", Arial, sans-serif | 28px |
| **Body** | "Microsoft YaHei", Arial, sans-serif | 18px |
| **Annotation** | "Microsoft YaHei", Arial, sans-serif | 14px |
| **Caption** | "Microsoft YaHei", Arial, sans-serif | 12px |

- **Formula policy**: `text-only` — 简单公式用 Unicode / 文字表达，无需 LaTeX PNG 渲染
- **Font install required**: 否 — 全部栈以 `Microsoft YaHei` 为主，Windows 预装；回退 `Arial, sans-serif`

## V. Layout Pattern

Swiss-minimal 风格：网格对齐、大留白、薄分割线、无圆角→小圆角卡片（r=4px）、锐利边界。

每个内容页遵循：
- 页眉区：标题 + 蓝色细线分隔（y=60-100）
- 内容区：卡片网格或左右分栏（y=110-620）
- 页脚区：页码（y=680-700）

## VI. Icon Usage

- **Library**: `tabler-outline` (stroke=2)
- **Inventory**: building-factory, robot, camera, brain, chart-bar, hierarchy, target, arrow-right, check, flask

## VII. Visualization Plan

本 PPT 不使用外部 chart 模板。可视化内容以手写 SVG 为主，含：
- 流程图（数据采集流程）
- 架构图（已生成的 `mvfact-full-figure.svg` 可引用为参考，但PPT中重新手写简化版）
- 表格（横向对比矩阵）
- 时间线（甘特图式）

## VIII. Image Resource List

| ID | Description | Acquire Via | Status |
|----|-------------|-------------|--------|
| (none) | 无外部图片需求 | `none` | — |

## IX. Content Outline

| Page | Title | Rhythm | Description |
|------|-------|--------|-------------|
| P01 | 增强视觉感知的ACT算法在机械臂装配任务研究 | anchor | 封面：课题名、陈绮颖、暨南大学、导师李德平 |
| P02 | 目录 | anchor | 四段式目录 |
| P03 | 研究背景与问题 | breathing | 透明杯抓取三大挑战 + 模仿学习泛化不足 |
| P04 | 实验平台与数据采集 | dense | 硬件照片位置（占位）+ 数据采集流程 + 数据集规格 |
| P05 | 数据集标注 | dense | CVAT标注 + SAM2 mask生成流程 + 标注统计 |
| P06 | M-VF-ACT 模型架构总览 | anchor | 简化架构图 + 核心创新标注 |
| P07 | 创新1：Mask-Guided Perception | dense | Mask Decoder设计 + 与FCOS并行关系 + 推理行为 |
| P08 | 检测+Mask 分支详解 | dense | FPN三路输出 → FCOS + Mask → 损失函数 |
| P09 | 实验计划：横向对比矩阵 | dense | 6组核心实验表格 + 评价指标 |
| P10 | 预期结论与决策门 | breathing | 根据实验结果 → 创新2/3/4优先级决策 |
| P11 | 时间线 | breathing | Phase 1-4 甘特图式时间线 |
| P12 | 感谢聆听 Q&A | anchor | 结束页 |

## X. Speaker Notes

每页提供简短的演讲提示（中文，与导师口述场景一致），重点页提供：
- 过渡语（"接下来看..."）
- 关键数字提醒（"90个episode, 120,000步训练"）
- 反问预案（"如果Mask无效怎么办？→ 看P10决策门"）

## XI. Tech Constraints

- SVG 仅用 PPT 安全子集：无 `<foreignObject>`, `<style>`, CSS class, `rgba()`, 动画
- 颜色使用 HEX 值
- 圆角用 `rx/ry`（非 clipPath）
- 文本用 `<text>` + `<tspan>`, 显式 `x/y` 定位
- 图标通过 `<use data-icon="...">` 占位，`finalize_svg.py` 处理嵌入
