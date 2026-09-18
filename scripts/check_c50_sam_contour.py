"""Compare user contour conditioning over a selected episode20 frame window."""
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from sam2.build_sam import build_sam2_video_predictor

from prepare_c50_mask_preview import boxes, extract_frames


def contour_mask(markup):
    """Interpolate left/right red outline intersections; not pixel ground truth."""
    pixels = np.asarray(markup)
    red = (pixels[:,:,0] > 180) & (pixels[:,:,1] < 100) & (pixels[:,:,2] < 100)
    yy, xx = np.nonzero(red)
    if len(xx) < 20:
        raise ValueError("Too few red outline pixels")
    rows, left, right = [], [], []
    for y in np.unique(yy):
        xs = xx[yy == y]
        if xs.max()-xs.min() >= 12:
            rows.append(int(y)); left.append(int(xs.min())); right.append(int(xs.max()))
    if len(rows) < 5:
        raise ValueError("Cannot infer two-sided contour")
    mask = np.zeros((480,640),dtype=bool)
    for y in range(int(yy.min()),int(yy.max())+1):
        x1 = int(round(np.interp(y,rows,left)))
        x2 = int(round(np.interp(y,rows,right)))
        mask[y,x1:x2+1] = True
    return mask, {"red_pixels":len(xx),"interpolation_rows":rows,
                  "method":"Per-row red-outline span, interpolated; approximate and unverified"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--start-frame", type=int, default=533)
    parser.add_argument("--end-frame", type=int, default=543)
    parser.add_argument("--modes", nargs="+", choices=["points_v2", "contour_sam_refine", "contour_seed_track"],
                        default=["points_v2", "contour_sam_refine", "contour_seed_track"])
    args = parser.parse_args()
    if not 0 <= args.start_frame <= 538 <= args.end_frame <= 717:
        raise ValueError("Frame window must contain seed538 and stay inside episode20")
    length = args.end_frame-args.start_frame+1
    seed_index = 538-args.start_frame
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[1]
    phase = root / "outputs/mask_inject_phase3_20260918"
    source = phase / "occlusion_review"
    markup = Image.open(source / "user_markup_ep020_f000538_v2.png").convert("RGB")
    raw = Image.open(source / "episode_020_frame_000538_raw.png").convert("RGB")
    if markup.size != raw.size or raw.size != (640,480):
        raise ValueError("Markup dimensions must match raw frame")
    seed, details = contour_mask(markup)
    points = json.loads((phase / "sam_prompt_check_ep020_f000538_v2/results.json").read_text())
    record = json.loads((phase / "mask_preview_framewise/episode_020.json").read_text())
    dataset = next(root.glob("*/formal1_C50_nomaster"))
    labels = boxes(dataset,20)[0]
    fps = json.loads((dataset / "meta/info.json").read_text())["fps"]
    output = args.output or phase / "sam_contour_ep020_533_543"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir()
    Image.fromarray(seed.astype(np.uint8)*255).save(output / "approximate_seed.png")
    frames = output / "input_frames"
    frames.mkdir()
    extract_frames(Path(record["video"]),record["video_from_timestamp"]+args.start_frame/fps,length,fps,frames)
    predictor = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_l.yaml",
        str(root / "checkpoints/sam2.1_hiera_large.pt"),device="cuda",apply_postprocessing=False)
    original_bypass = predictor.use_mask_input_as_output_without_sam
    report = {"episode":20,"frames":list(range(args.start_frame,args.end_frame+1)),"seed_frame":538,
              "contour_extraction":details,"seed_area":int(seed.sum()),
              "default_mask_input_bypasses_sam_decoder":bool(original_bypass),
              "formal_training_ready":False,"review_status":"pending",
              "warning":"Seed is an approximate user-outline interpolation, not pixel ground truth",
              "results":{}}
    predictions = {}
    box = labels[538]
    x1,y1,x2,y2 = box
    background = [[x1-25,(y1+y2)/2],[x2+25,(y1+y2)/2],[(x1+x2)/2,y1-25],[(x1+x2)/2,y2+25]]
    background = [p for p in background if 0 <= p[0] < 640 and 0 <= p[1] < 480]
    with torch.inference_mode():
        state = predictor.init_state(video_path=str(frames),offload_video_to_cpu=True,offload_state_to_cpu=True)
        for name in args.modes:
            predictor.reset_state(state)
            predictor.use_mask_input_as_output_without_sam = name != "contour_sam_refine"
            if name == "points_v2":
                coords = background+points["positive_points"]+points["negative_points"]
                point_labels = [0]*len(background)+[1]*len(points["positive_points"])+[0]*len(points["negative_points"])
                predictor.add_new_points_or_box(state,frame_idx=seed_index,obj_id=1,box=box,
                    points=np.asarray(coords,dtype=np.float32),labels=np.asarray(point_labels,dtype=np.int32))
            else:
                predictor.add_new_mask(state,frame_idx=seed_index,obj_id=1,mask=seed)
            masks = np.zeros((length,480,640),dtype=np.uint8)
            seen = np.zeros(length,dtype=bool)
            for reverse in [False,True]:
                for index,objects,logits in predictor.propagate_in_video(state,start_frame_idx=seed_index,reverse=reverse):
                    if objects != [1] or not 0 <= index < length or (seen[index] and index != seed_index):
                        raise ValueError("Unexpected propagation result")
                    masks[index] = (logits[0,0] > 0).cpu().numpy().astype(np.uint8)
                    seen[index] = True
            if not seen.all():
                raise ValueError("Incomplete propagation")
            area = masks.sum(axis=(1,2)).astype(float)
            intersection = (masks[seed_index].astype(bool) & seed).sum()
            union = (masks[seed_index].astype(bool) | seed).sum()
            spill = []
            for index,mask in enumerate(masks):
                bx1,by1,bx2,by2 = labels[index+args.start_frame]
                inside = mask[max(0,int(np.floor(by1))):min(480,int(np.ceil(by2))+1),
                              max(0,int(np.floor(bx1))):min(640,int(np.ceil(bx2))+1)].sum()
                spill.append(float((area[index]-inside)/max(area[index],1)))
            report["results"][name] = {"areas":area.tolist(),
                "empty_frames":(np.flatnonzero(area == 0)+args.start_frame).tolist(),
                "area_jumps_over_50pct":(np.flatnonzero(np.abs(np.diff(area)) > 0.5*np.maximum(area[:-1],1))+args.start_frame+1).tolist(),
                "outside_box_fraction":spill,"mean_outside_box_fraction":float(np.mean(spill)),
                "agreement_with_approximate_seed_not_accuracy":float(intersection/max(union,1))}
            predictions[name] = masks
            np.savez_compressed(output / f"{name}.npz",masks=masks,valid=seen)
    predictor.use_mask_input_as_output_without_sam = original_bypass
    sample_indices = {0,seed_index,length-1}
    if length > 11:
        sample_indices.update(range(0,length,120))
        sample_indices.update(frame-args.start_frame for frame in [360,400,541,543,547]
                              if args.start_frame <= frame <= args.end_frame)
        for result in report["results"].values():
            for frame in result["area_jumps_over_50pct"][:8]:
                sample_indices.update(i-args.start_frame for i in [frame-1,frame,frame+1]
                                      if args.start_frame <= i <= args.end_frame)
            sample_indices.update(int(i) for i in np.argsort(result["outside_box_fraction"])[-3:])
    report["sample_frames"] = [index+args.start_frame for index in sorted(sample_indices)]
    for index in sorted(sample_indices):
        pixels = np.asarray(Image.open(frames / f"{index:06d}.jpg")).copy()
        sheet = Image.new("RGB",(640*(1+len(predictions)),510),"white")
        for column,(name,mask) in enumerate([("APPROXIMATE CONTOUR" if index == seed_index else "RAW FRAME",seed if index == seed_index else None)]
                +[(name,masks[index]) for name,masks in predictions.items()]):
            rgb = pixels.copy()
            if mask is not None:
                fg = mask.astype(bool)
                rgb[fg] = (0.65*rgb[fg]+0.35*np.array([0,255,0])).astype(np.uint8)
            panel = Image.fromarray(rgb)
            ImageDraw.Draw(panel).rectangle(labels[index+args.start_frame],outline="yellow",width=1)
            sheet.paste(panel,(column*640,30))
            ImageDraw.Draw(sheet).text((column*640+8,8),f"frame={index+args.start_frame} {name}",fill="black")
        sheet.save(output / f"frame_{index+args.start_frame:06d}_comparison.jpg",quality=95)
    (output / "results.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({"episode":20,"frames_complete":length,"output":str(output),
        "results":{name:{key:result[key] for key in ["empty_frames","area_jumps_over_50pct","mean_outside_box_fraction"]}
                   for name,result in report["results"].items()}}))


if __name__ == "__main__":
    main()
