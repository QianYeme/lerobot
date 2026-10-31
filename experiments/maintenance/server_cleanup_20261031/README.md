# 2026-10-31 server cleanup

User authorized historical intermediate checkpoint/cache deletion. No formal P4.9 training existed during cleanup.

- Deleted52 historical intermediate resume.pt files; every historical run latest and every80k candidate retained (18 resume files in historical formal6 track).
- Removed pip cache, HuggingFace Xet transport cache and old purchase-document node_modules. DINO assets, environments, datasets, formal r5/r6 assets, all logs/videos and robot exports preserved.
- Deleted files6489; logical bytes44,500,238,842. Data disk actually freed43,913,572,352 bytes (40.90GiB); free63,644,782,592 bytes (59.28GiB). System disk additionally freed603,074,560 bytes.
- All retained historical resume SHA checks passed; all256 original r5 manifest assets rehashed unchanged.
- deletion_plan.json lists each deleted path/size/SHA/reason, plus retained resume path/SHA. Deletion is irreversible unless a separate backup exists; no claim that intermediate resume files were uploaded elsewhere.
- Final historical states were retained even for nonselected variants to preserve provenance; no unsupported assumption that a whole model was eliminated.

The report describes cleanup time; subsequent P4.9 checkpoints will consume free space.
