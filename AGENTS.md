# Project Agent Instructions

## Project goal

Develop and evaluate ACT/ACTDet policies for an SO-101 robot performing accurate, smooth pick–move–place of a transparent, water-filled cup.

## Required context

- For project status, experiment planning, model/data changes, training, deployment, or real-robot work, read `workflow/STATUS.md` first.
- For L2–L4 work, read the relevant sections of `workflow/leorbot_workflow.md` before acting.
- Use `docc/日志/工作日志.md` for chronology and follow the report/guide links in `workflow/STATUS.md` for detailed evidence.
- For L0–L1 tasks, read only the files relevant to the request; do not load the entire history without need.

## Working rules

- State assumptions and success criteria before non-trivial implementation.
- Preserve unrelated user changes in a dirty worktree; do not reformat or clean them.
- Do not silently change dataset splits, state/action schema, camera semantics, preprocessing, metrics, model comparison, or deployment parameters.
- Distinguish code smoke tests, offline teacher-forced evidence, and real-robot evidence.
- Unknown pixels are not background; occlusion is not a negative label; interpolation is not human ground truth.
- Run targeted regression tests and `git diff --check`; model/data changes also need a real-data smoke test before full training.
- Never report a check as passed unless it actually ran successfully.
- After material project changes, update `workflow/STATUS.md` and append `docc/日志/工作日志.md` as appropriate.

## Authority and safety

- Do not commit, push, publish models, launch a new expensive training campaign, or operate the real robot unless the user requested that action.
- Use explicit file staging; never use `git add .` in a dirty worktree.
- Keep credentials out of repository files and logs.
- Real-robot work requires the deployment and safety gates in the workflow; stop on collision, loss-of-control, or spill risk.
