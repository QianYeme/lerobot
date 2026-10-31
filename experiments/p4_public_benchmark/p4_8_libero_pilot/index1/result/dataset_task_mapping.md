# Dataset ↔ simulator task mapping

Simulator suite=libero_object, task_id=0; exact normalized instruction: **pick up the alphabet soup and place it in the basket**.
Public dataset task_index=24,44 episodes,6867 frames; all44 IDs in selected_task.json. Do not treat simulator task0 as dataset task_index0.
Selected before either model's performance: first canonical simulator task, exact language match to actual tasks.parquet and frame task_index. Every episode has one task; per-episode lengths match meta. Sanity episodes813/1035/1245 each first/middle/last frame.
Public data revision a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4; simulator assets0b3ea86be5fe169d0fd036ae63d1070ec09e90f6. Whole table coverage1693 episodes/273465 frames/40 tasks was audited; only selected-task videos were fetched for sanity.
No Train/Dev split has been approved or used for model training. The proposed split below is not an executed scientific variable.
