"""Extract three user red-dot annotations; no automatic water recognition."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from prepare_c50_mask_preview import boxes


def main():
    root = Path(__file__).resolve().parents[1]
    pilot = root / "outputs/water_center_pilot_20260918"
    output = pilot / "annotations.json"
    if output.exists():
        raise FileExistsError(output)
    dataset = next(root.glob("*/formal1_C50_nomaster"))
    cup_boxes, xml = boxes(dataset,20)
    entries = []
    for frame, extension in [(0,"jpg"),(538,"png"),(717,"jpg")]:
        path = pilot / "marked" / f"episode020_frame{frame:06d}.{extension}"
        image = Image.open(path).convert("RGB")
        if image.size != (640,480):
            raise ValueError("Original 640x480 coordinates required")
        pixels = np.asarray(image).astype(np.int16)
        red = (pixels[:,:,0] > 140) & (pixels[:,:,0]-pixels[:,:,1] > 60) & (pixels[:,:,0]-pixels[:,:,2] > 60)
        yy,xx = np.nonzero(red)
        if not 3 <= len(xx) <= 100 or np.ptp(xx) > 12 or np.ptp(yy) > 12:
            raise ValueError(f"Expected one small red marker in {path.name}")
        x,y = float(xx.mean()),float(yy.mean())
        x1,y1,x2,y2 = cup_boxes[frame]
        if not x1 <= x <= x2 or not y1 <= y <= y2:
            raise ValueError(f"Marker outside cup box at frame {frame}")
        entries.append({"episode":20,"frame":frame,"camera":"top","point_xy":[x,y],
            "red_marker_pixels":len(xx),"cup_box":[x1,y1,x2,y2],
            "box_relative_xy":[(x-x1)/(x2-x1),(y-y1)/(y2-y1)],
            "offset_from_box_center_px":[x-(x1+x2)/2,y-(y1+y2)/2],
            "marked_image_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "source":"User red-dot label; extraction is marker detection, not water detection"})
    report = {"target":"User-defined visible water reference center",
        "semantic_status":"Not independently established as physical water-surface center rather than cup-bottom projection",
        "coordinate_system":"Original 640x480 image, x right / y down",
        "episode":20,"training_only":True,"frames":entries,"formal_training_ready":False,
        "annotation_xml_sha256":hashlib.sha256(xml.read_bytes()).hexdigest(),
        "warning":"Three frames do not establish tracking stability, localization accuracy, or grasp coordinates"}
    output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(entries,indent=2))


if __name__ == "__main__":
    main()
