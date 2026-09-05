#!/usr/bin/env python
"""枚举本机所有相机，每个可打开的序号拍一帧存到 outputs/camera_check/ 下。

用途：真机测试前确认哪个序号是「桌面全局视角」(填给 top)、哪个是「腕部视角」(填给 gripper)。
用法：python probe_cameras.py [起始序号] [结束序号)
"""

import sys

import cv2

START = int(sys.argv[1]) if len(sys.argv) > 1 else 0
END = int(sys.argv[2]) if len(sys.argv) > 2 else 8

out_dir = "outputs/camera_check"
import os

os.makedirs(out_dir, exist_ok=True)

for i in range(START, END):
    cap = cv2.VideoCapture(i)
    if not cap.isOpened():
        print(f"[{i}] 打不开 (无此相机)")
        continue
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"[{i}] 打开了但读不到帧")
        continue
    # 读一帧就退，某些相机首帧黑，多读几次取最后一张非黑
    cap = cv2.VideoCapture(i)
    last = frame
    for _ in range(10):
        ok, f = cap.read()
        if ok and f.mean() > 5:
            last = f
    cap.release()
    path = f"{out_dir}/probe_{i}.jpg"
    cv2.imwrite(path, last)
    h, w = last.shape[:2]
    b, g, r = last[:, :, 0].mean(), last[:, :, 1].mean(), last[:, :, 2].mean()
    print(f"[{i}] OK -> {path}  ({w}x{h}  BGR均值=({b:.0f},{g:.0f},{r:.0f}))")
