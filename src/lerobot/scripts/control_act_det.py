#!/usr/bin/env python

# Copyright 2026 QianYeme. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Minimal real-robot rollout for trained ACT / ACTDet checkpoints.

Reuses the record pipeline (robot + cameras + policy pre/post processors) but
does NOT write a dataset — the follower arm is controlled by the policy while
the human judges the 4-stage success per trial and records video externally.

The observation.state vector is read live from the follower: 5 arm joints +
gripper.pos + gripper.load + gripper.curr + master_gripper.pos (the last one is
a constant 0.0 without a leader arm, matching the policy-mode behavior of
lerobot-record).

Example:
```shell
python src/lerobot/scripts/control_act_det.py \
    --policy.path outputs/train/2026-08-08/12-34-49_act_det/checkpointsE9/last/pretrained_model \
    --robot.type=so_follower \
    --robot.port=/dev/ttyUSB0 \
    --robot.cameras='{top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, gripper: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}' \
    --dataset.repo_id formal1_B \
    --dataset.root /root/autodl-tmp/lerobot/lerobot-main/数据集/formal1_B \
    --dataset.single_task="Cup pick and place" \
    --dataset.num_episodes 10 \
    --dataset.episode_time_s 60 \
    --dataset.reset_time_s 15 \
    --dataset.fps 30
```

Keys: right arrow = end current episode early, Esc = stop. The camera names in
`--robot.cameras` must match the dataset's camera keys (`top`, `gripper`).
"""

import logging
import time
from pathlib import Path

import cv2

from lerobot.configs import parser
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.feature_utils import build_dataset_frame
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.processor import (
    RobotAction,
    make_default_processors,
)
from lerobot.scripts.lerobot_record import RecordConfig
from lerobot.robots import make_robot_from_config
from lerobot.utils.constants import OBS_STR
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    predict_action,
)
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say


@parser.wrap()
def control(cfg: RecordConfig):
    init_logging()

    if cfg.robot is None:
        raise ValueError("You need to provide a robot config, e.g. --robot.type=so_follower")
    if cfg.policy is None:
        raise ValueError("You need to provide a policy with --policy.path")

    robot = make_robot_from_config(cfg.robot)

    # Read-only dataset metadata: gives normalization stats and feature schema.
    ds_meta = LeRobotDatasetMetadata(cfg.dataset.repo_id, root=cfg.dataset.root)

    policy = make_policy(cfg.policy, ds_meta=ds_meta)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        dataset_stats=ds_meta.stats,
    )

    _, robot_action_processor, robot_observation_processor = make_default_processors()
    device = get_safe_torch_device(policy.config.device)

    robot.connect()

    # Camera sanity check: save one frame per camera so the operator can confirm
    # the policy is seeing the correct views (a wrong index_or_path fails silently
    # and the policy runs "blind", which looks identical to model collapse).
    obs = robot.get_observation()
    check_dir = Path("outputs/camera_check")
    check_dir.mkdir(parents=True, exist_ok=True)
    image_keys = [k for k in obs if k.startswith("observation.images.")]
    for key in image_keys:
        cam_name = key.removeprefix("observation.images.")
        path = check_dir / f"{cam_name}.jpg"
        cv2.imwrite(str(path), obs[key])
        logging.info("Camera frame saved: %s", path)
    if not is_headless():
        input(
            f"已保存 {len(image_keys)} 路相机画面到 {check_dir}。\n"
            "请确认 top=桌面全局视角、gripper=腕部视角且画面正常，按回车继续。"
        )

    listener, events = init_keyboard_listener()
    try:
        for episode in range(cfg.dataset.num_episodes):
            if events["stop_recording"]:
                break

            policy.reset()
            preprocessor.reset()
            postprocessor.reset()

            log_say(f"Episode {episode + 1}/{cfg.dataset.num_episodes} — policy running", cfg.play_sounds)
            start_episode_t = time.perf_counter()
            timestamp = 0
            loop_idx = 0
            prev_gripper_target = None
            prev_gripper_state = None
            while timestamp < cfg.dataset.episode_time_s:
                if events["exit_early"]:
                    events["exit_early"] = False
                    break

                start_loop_t = time.perf_counter()

                obs = robot.get_observation()
                obs_processed = robot_observation_processor(obs)

                observation_frame = build_dataset_frame(
                    ds_meta.features, obs_processed, prefix=OBS_STR
                )
                action_values = predict_action(
                    observation=observation_frame,
                    policy=policy,
                    device=device,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    use_amp=policy.config.use_amp,
                    task=cfg.dataset.single_task,
                    robot_type=robot.robot_type,
                )

                act_processed_policy: RobotAction = make_robot_action(
                    action_values, ds_meta.features
                )
                robot_action_to_send = robot_action_processor((act_processed_policy, obs))

                # ---- 夹爪诊断: 实际位置 / 模型目标 / 实际发送(可能被 max_relative_target 钳位) ----
                gripper_target = act_processed_policy.get("gripper.pos")
                gripper_actual = obs.get("gripper.pos")
                sent_action = robot.send_action(robot_action_to_send)
                gripper_sent = sent_action.get("gripper.pos")

                if gripper_target is not None:
                    # 夹爪"张开/闭合"状态(阈值对齐 demo: <20≈闭, >35≈张, 中间≈抓持)
                    state = "闭" if gripper_target < 20 else ("张" if gripper_target > 35 else "中")
                    if prev_gripper_state is not None and state != prev_gripper_state:
                        # 只对"张→闭"(抓取) 和 "闭→张"(放下) 这两类关键转换报警
                        if prev_gripper_state == "张" and state in ("闭", "中"):
                            logging.info(
                                f"[grasp] ★闭合抓取: target {prev_gripper_target:.1f} -> {gripper_target:.1f} "
                                f"(t={timestamp:.1f}s)"
                            )
                        elif prev_gripper_state == "闭" and state in ("张", "中"):
                            logging.info(
                                f"[grasp] ★张开放下: target {prev_gripper_target:.1f} -> {gripper_target:.1f} "
                                f"(t={timestamp:.1f}s)"
                            )
                    prev_gripper_state = state
                    prev_gripper_target = gripper_target

                if loop_idx % 15 == 0:
                    logging.info(
                        f"[gripper] t={timestamp:.1f}s 实际={gripper_actual:.1f} "
                        f"目标={gripper_target:.1f} 发送={gripper_sent:.1f}"
                    )
                loop_idx += 1

                dt_s = time.perf_counter() - start_loop_t
                sleep_time_s = 1 / cfg.dataset.fps - dt_s
                if sleep_time_s < 0:
                    logging.warning(
                        f"Control loop is running slower ({1 / dt_s:.1f} Hz) than the target "
                        f"FPS ({cfg.dataset.fps} Hz). Robot control might be unstable."
                    )
                precise_sleep(max(sleep_time_s, 0.0))

                timestamp = time.perf_counter() - start_episode_t

            # Reset interval: no action is sent, motors hold the last target.
            if episode < cfg.dataset.num_episodes - 1 and not events["stop_recording"]:
                log_say("Reset the environment and place the cup at the next preset pose",
                        cfg.play_sounds)
                reset_start = time.perf_counter()
                while time.perf_counter() - reset_start < cfg.dataset.reset_time_s:
                    if events["exit_early"]:
                        events["exit_early"] = False
                        break
                    precise_sleep(0.5)
    finally:
        log_say("Stopping", cfg.play_sounds, blocking=True)
        if robot.is_connected:
            robot.disconnect()
        if not is_headless() and listener:
            listener.stop()
        log_say("Exiting", cfg.play_sounds)


def main():
    register_third_party_plugins()
    control()


if __name__ == "__main__":
    main()
