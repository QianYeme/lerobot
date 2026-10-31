# r5 completed run: artifact locations

This is a completed pilot, not an instruction to launch another campaign.

- Server project: /root/autodl-tmp/lerobot-main.
- Executed isolated trainer/evaluator: index5/workspace/scripts; immutable object module: index5/workspace/object_conditioning.py. Source copies are delivered under result/tools; preparation exporters/builders and read-only postrun audit are also included there.
- Native ACT source: inherited index1/result/workspace/lerobot/src, official commit8fff0fde7c79f23a93d845d1a50e985de01f8b8a.
- Python: /root/autodl-tmp/p48_isolated_env/bin/python; inherited Py3.12.3/Torch2.8.0+cu128/MuJoCo3.3.6/robosuite1.4.0. Existing G1 environment records remain in index1. There were no r5 package installs or base environment changes.
- LIBERO data revision a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4 at /root/autodl-tmp/p48_data/lerobot_libero; simulator assets remain pinned from G1. Train35/Dev9 IDs and stats are delivered.
- Full resume checkpoints: result/runs/{act,legacy,fixed}/checkpoints/{005000,010000}/resume.pt. They include model, optimizer, RNG, epoch/next cursor, split/bank/stats hashes. checkpoint_manifest.csv identifies every formal file uniquely. Engineering100-step files stay separate under result/smoke and are not formal policy models.
- Large RGB caches and checkpoints are retained only server-side and listed with exact size/SHA in artifact_manifest.json. They are intentionally absent from ZIP. Source DINO weights are also contained in P1 checkpoint model state; pretrained cache SHA is in prototype_numeric_validation.json. dinov2_source_snapshot.zip preserves the exact available Python source and LICENSE; unknown upstream git revision is not invented.
- All30 rollout videos and JSON traces are included. Success comes from the environment predicate; videos show pre-action frames because the native wrapper resets immediately on termination. Failure phase is UNKNOWN without an oracle.
- The original server auto ZIP and receipt are preserved. REVIEWED ZIP adds executor checks and mechanism findings without rerunning policies or changing original evidence. Its review_artifact_manifest.json covers packaged additions and preserves original large-file pointers.
- Independent Builder/Red-Team review is still pending; another budget/task/seed/architecture needs a new decision.
