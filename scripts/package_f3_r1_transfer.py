"""Credential-free new r1 contract/scripts, preserving original F3 r0."""
import tarfile
from pathlib import Path

root=Path(__file__).resolve().parents[1]
base=root/'experiments/force/f3_real_robot_multi_model_comparison/index1'
with tarfile.open(root/'outputs/f3_r1_inputs.tar','w') as archive:
    for name in ['publish_f3_r1_bundle.py','verify_f3_multimodel_bundle.py','publish_f3_r1.sh']:
        archive.add(root/'scripts'/name,arcname=name)
    archive.add(base/'TRANSFER_GUIDE.md',arcname='TRANSFER_GUIDE.md')
    archive.add(base/'decision_F3_REAL_ROBOT_MULTI_MODEL_COMPARISON_r1.md',arcname='decision.md')
