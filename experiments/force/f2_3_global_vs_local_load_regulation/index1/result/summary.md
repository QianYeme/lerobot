# F2.3 — LOCAL_LOAD_REGULATION_PROMISING

Completed matched two local MLP runs; C0/global checkpoints reused, not retrained.

{
  "C0": {
    "primary": 0.12639093399047852,
    "physical_gripper_mae": 3.148035908032627,
    "physical_arm_mae": 2.965693893646326,
    "normalized_all_l1": 0.12033772468566895,
    "frames": 5748
  },
  "C1-no-load": {
    "primary": 0.08885254710912704,
    "physical_gripper_mae": 2.213062210350048,
    "physical_arm_mae": 3.484145663755488,
    "normalized_all_l1": 0.13040538132190704,
    "frames": 5748
  },
  "C1-load": {
    "primary": 0.05769287794828415,
    "physical_gripper_mae": 1.4369640758956177,
    "physical_arm_mae": 3.484145663755488,
    "normalized_all_l1": 0.12521210312843323,
    "frames": 5748
  }
}

Single-seed exploratory teacher-forced evidence. No calibrated force or robot success claim. Final sealed. Stop; external Builder/Red-Team review pending.
