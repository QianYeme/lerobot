"""Isolated remote source overlay for NO-hardware control validation."""
import tarfile
from pathlib import Path

root=Path(__file__).resolve().parents[1]
names=['src/lerobot/scripts/control_f3.py','src/lerobot/policies/act/f3_runtime.py',
    'src/lerobot/policies/act/local_load_regulator.py','src/lerobot/utils/f3_control_protocol.py',
    'src/lerobot/policies/act/configuration_act.py','src/lerobot/policies/act/modeling_act.py',
    'scripts/smoke_f3_control.py','scripts/smoke_f3_control_mock.py',
    'tests/scripts/test_f3_control_cli.py','tests/scripts/test_f3_control_protocol.py']
with tarfile.open(root/'outputs/f3_control_validation.tar','w') as archive:
    for name in names:archive.add(root/name,arcname='source/'+name)
