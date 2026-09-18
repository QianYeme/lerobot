"""Assess existing SAM mask centroids against three user water-reference points."""
import json
from pathlib import Path

import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    pilot = root / "outputs/water_center_pilot_20260918"
    output = pilot / "candidate_assessment.json"
    if output.exists():
        raise FileExistsError(output)
    labels = json.loads((pilot / "annotations.json").read_text())
    with np.load(root / "outputs/mask_inject_phase3_20260918/sam_contour_ep020_full/contour_seed_track.npz") as archive:
        masks = archive["masks"]
    if masks.shape != (718,480,640):
        raise ValueError("Expected full episode20 masks")
    centers = []
    for frame,mask in enumerate(masks):
        yy,xx = np.nonzero(mask)
        if not len(xx):
            raise ValueError(f"Empty candidate mask at {frame}")
        centers.append([float(xx.mean()),float(yy.mean())])
    centers = np.asarray(centers)
    comparisons = []
    for label in labels["frames"]:
        frame = label["frame"]
        error = centers[frame]-np.asarray(label["point_xy"])
        comparisons.append({"frame":frame,"user_xy":label["point_xy"],"candidate_xy":centers[frame].tolist(),
                            "error_px":float(np.linalg.norm(error))})
    displacement = np.linalg.norm(np.diff(centers,axis=0),axis=1)
    ranked = np.argsort(displacement)[-10:][::-1]+1
    report = {"candidate":"Centroid of existing contour-seeded SAM tracking mask; not automatic water detection",
              "comparisons":comparisons,"centers_xy":centers.tolist(),
              "displacement_px":{"median":float(np.median(displacement)),"p95":float(np.percentile(displacement,95)),
                                  "max":float(displacement.max())},
              "largest_displacement_frames":[{"frame":int(frame),"displacement_px":float(displacement[frame-1])} for frame in ranked],
              "formal_training_ready":False,
              "warning":"Only 3 labels in one episode; seed538 is not independent. Motion displacement includes real cup motion and cannot establish tracking error."}
    output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({key:report[key] for key in ["comparisons","displacement_px","largest_displacement_frames","warning"]},indent=2))


if __name__ == "__main__":
    main()
