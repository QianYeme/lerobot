"""Build credential-free server inputs for the F3 migration artifact export."""
import tarfile
from pathlib import Path

root=Path(__file__).resolve().parents[1]
base=root/'experiments/force/f3_real_robot_local_regulator_feasibility/index1'
files={
    'publish_f3_bundle.py':root/'scripts/publish_f3_bundle.py',
    'verify_f3_bundle.py':root/'scripts/verify_f3_bundle.py',
    'local_load_regulator.py':root/'src/lerobot/policies/act/local_load_regulator.py',
    'pyproject.toml':root/'pyproject.toml',
    'TRANSFER_GUIDE.md':base/'TRANSFER_GUIDE.md',
    'decision.md':base/'decision_F3_REAL_ROBOT_LOCAL_REGULATOR_FEASIBILITY.md',
}
with tarfile.open(root/'outputs/f3_transfer_inputs.tar','w') as archive:
    for name,path in files.items():archive.add(path,arcname=name)
