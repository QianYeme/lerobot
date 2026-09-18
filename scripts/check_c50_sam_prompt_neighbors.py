"""Propagate reviewed frame538 prompts over a small 533..543 window."""
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from sam2.build_sam import build_sam2_video_predictor

from prepare_c50_mask_preview import boxes, extract_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-results", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[1]
    phase = root / "outputs/mask_inject_phase3_20260918"
    points = json.loads((args.seed_results or phase / "sam_prompt_check_ep020_f000538/results.json").read_text())
    record = json.loads((phase / "mask_preview_framewise/episode_020.json").read_text())
    dataset = next(root.glob("*/formal1_C50_nomaster"))
    labels = boxes(dataset, 20)[0]
    fps = json.loads((dataset / "meta/info.json").read_text())["fps"]
    output = args.output or phase / "sam_prompt_neighbors_ep020_533_543"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir()
    frames = output / "input_frames"
    frames.mkdir()
    extract_frames(Path(record["video"]), record["video_from_timestamp"]+533/fps, 11, fps, frames)
    box = labels[538]
    x1,y1,x2,y2 = box
    background = [[x1-25,(y1+y2)/2],[x2+25,(y1+y2)/2],[(x1+x2)/2,y1-25],[(x1+x2)/2,y2+25]]
    background = [point for point in background if 0 <= point[0] < 640 and 0 <= point[1] < 480]
    predictor = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_l.yaml",
        str(root / "checkpoints/sam2.1_hiera_large.pt"), device="cuda", apply_postprocessing=False)
    report = {"episode":20,"frames":list(range(533,544)),"seed_frame":538,
              "formal_training_ready":False,"warning":"Local-window prompt ablation, not full-episode accuracy",
              "results":{}}
    predictions = {}
    with torch.inference_mode():
        state = predictor.init_state(video_path=str(frames),offload_video_to_cpu=True,offload_state_to_cpu=True)
        for name,coords,point_labels in [
            ("box_background",background,[0]*len(background)),
            ("user_points",background+points["positive_points"]+points["negative_points"],
             [0]*len(background)+[1]*len(points["positive_points"])+[0]*len(points["negative_points"]))]:
            predictor.reset_state(state)
            predictor.add_new_points_or_box(state,frame_idx=5,obj_id=1,box=box,
                points=np.asarray(coords,dtype=np.float32),labels=np.asarray(point_labels,dtype=np.int32))
            masks = np.zeros((11,480,640),dtype=np.uint8)
            seen = np.zeros(11,dtype=bool)
            for reverse in [False,True]:
                for index,objects,logits in predictor.propagate_in_video(state,start_frame_idx=5,reverse=reverse):
                    if objects != [1] or not 0 <= index < 11:
                        raise ValueError("Unexpected propagation result")
                    if seen[index] and index != 5:
                        raise ValueError("Unexpected duplicate frame")
                    masks[index] = (logits[0,0] > 0).cpu().numpy().astype(np.uint8)
                    seen[index] = True
            if not seen.all():
                raise ValueError("Incomplete local propagation")
            area = masks.sum(axis=(1,2)).astype(float)
            jumps = np.flatnonzero(np.abs(np.diff(area)) > 0.5*np.maximum(area[:-1],1))+534
            report["results"][name] = {"areas":area.tolist(),"area_jumps_over_50pct":jumps.tolist(),
                                       "empty_frames":(np.flatnonzero(area == 0)+533).tolist()}
            np.savez_compressed(output/f"{name}.npz",masks=masks,valid=seen)
            predictions[name] = masks
    sheet = Image.new("RGB",(1280,11*510),"white")
    for index in range(11):
        raw = np.asarray(Image.open(frames/f"{index:06d}.jpg")).copy()
        for column,name in enumerate(predictions):
            pixels = raw.copy()
            foreground = predictions[name][index].astype(bool)
            pixels[foreground] = (0.65*pixels[foreground]+0.35*np.array([0,255,0])).astype(np.uint8)
            panel = Image.fromarray(pixels)
            ImageDraw.Draw(panel).rectangle(labels[index+533],outline="red",width=1)
            sheet.paste(panel,(column*640,index*510+30))
            ImageDraw.Draw(sheet).text((column*640+8,index*510+8),f"frame={index+533} {name}",fill="black")
    sheet.save(output/"comparison.jpg",quality=95)
    (output/"results.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
