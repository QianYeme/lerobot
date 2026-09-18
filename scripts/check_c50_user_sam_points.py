"""Single-frame SAM prompt ablation from user markup; not training-label approval."""
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from sam2.build_sam import build_sam2_video_predictor

from prepare_c50_mask_preview import boxes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[1]
    dataset = next(root.glob("*/formal1_C50_nomaster"))
    source = root / "outputs/mask_inject_phase3_20260918/occlusion_review"
    output = args.output or root / "outputs/mask_inject_phase3_20260918/sam_prompt_check_ep020_f000538"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    raw = Image.open(source / "episode_020_frame_000538_raw.png").convert("RGB")
    if raw.size != (640, 480):
        raise ValueError("Expected original image dimensions")
    box = boxes(dataset, 20)[0][538]
    positive = [[137, 247]]
    negative = [[126, 260], [158, 258]]
    if args.prompt_file:
        prompt = json.loads(args.prompt_file.read_text())
        positive, negative = prompt["positive_points"], prompt["negative_points"]
    if not positive or any(len(point) != 2 or not all(isinstance(v, int) for v in point)
                           or not 0 <= point[0] < 640 or not 0 <= point[1] < 480
                           for point in positive + negative):
        raise ValueError("Expected integer original-image points with at least one positive")
    x1, y1, x2, y2 = box
    background = [[x1-25, (y1+y2)/2], [x2+25, (y1+y2)/2],
                  [(x1+x2)/2, y1-25], [(x1+x2)/2, y2+25]]
    background = [point for point in background if 0 <= point[0] < 640 and 0 <= point[1] < 480]
    predictor = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_l.yaml",
        str(root / "checkpoints/sam2.1_hiera_large.pt"), device="cuda", apply_postprocessing=False)
    output.mkdir(parents=True)
    frames = output / "input_frames"
    frames.mkdir()
    raw.save(frames / "000000.jpg", quality=95)
    results = {}
    masks = []
    with torch.inference_mode():
        state = predictor.init_state(video_path=str(frames), offload_video_to_cpu=True, offload_state_to_cpu=True)
        for name, points, labels in [
                ("box_background", background, [0]*len(background)),
                ("user_points", background+positive+negative, [0]*len(background)+[1]*len(positive)+[0]*len(negative))]:
            predictor.reset_state(state)
            _, objects, logits = predictor.add_new_points_or_box(state, frame_idx=0, obj_id=1,
                box=box, points=np.asarray(points, dtype=np.float32), labels=np.asarray(labels, dtype=np.int32))
            if objects != [1]:
                raise ValueError("Unexpected SAM objects")
            mask = (logits[0, 0] > 0).cpu().numpy().astype(np.uint8)
            masks.append(mask)
            np.savez_compressed(output / f"{name}.npz", masks=mask[None])
            results[name] = {"area": int(mask.sum()),
                            "positive_points_in_mask": [bool(mask[y,x]) for x,y in positive],
                            "negative_points_in_mask": [bool(mask[y,x]) for x,y in negative]}
    sheet = Image.new("RGB", (1920,510), "white")
    for column, (name, mask) in enumerate(zip(["INPUT + SELECTED POINTS", "BOX + BACKGROUND", "ADD USER CUP / GRIPPER POINTS"], [None]+masks)):
        pixels = np.asarray(raw).copy()
        if mask is not None:
            foreground = mask.astype(bool)
            pixels[foreground] = (0.65*pixels[foreground]+0.35*np.array([0,255,0])).astype(np.uint8)
        panel = Image.fromarray(pixels)
        draw = ImageDraw.Draw(panel)
        draw.rectangle(box, outline="yellow", width=1)
        for points, color in [(positive,"red"),(negative,"blue")]:
            for x,y in points:
                draw.ellipse((x-3,y-3,x+3,y+3), fill=color)
        sheet.paste(panel,(column*640,30))
        ImageDraw.Draw(sheet).text((column*640+8,8),name,fill="black")
    sheet.save(output / "comparison.jpg", quality=95)
    report = {"episode":20,"frame":538,"positive_points":positive,"negative_points":negative,
              "point_source":"Assistant selected representative pixels from user red cup outline / blue gripper markup",
              "sam_input":"Unmarked original image re-encoded JPEG quality95; not user painted image",
              "results":results,"review_status":"pending_visual_review","formal_training_ready":False,
              "warning":"Point satisfaction is not pixel accuracy or temporal stability"}
    (output / "results.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
