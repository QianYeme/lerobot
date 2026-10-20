"""Freeze delivery runtime identity, preserving the already-running training source."""
import difflib
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

BASE=Path('/root/autodl-tmp/F2_2_r0_20261019')
PARENT=Path('/root/autodl-tmp/F2_1_r1_20261019')


def main():
    inputs=BASE/'inputs'
    assert not (inputs/'source_hashes.json').exists(), 'Do not silently refreeze runtime'
    patches=[]
    for name in ['src/lerobot/policies/act/configuration_act.py','src/lerobot/policies/act/modeling_act.py']:
        patches.extend(difflib.unified_diff((PARENT/'source'/name).read_text().splitlines(keepends=True),
            (BASE/'source'/name).read_text().splitlines(keepends=True),fromfile='F2.1/'+name,tofile='F2.2/'+name))
    (inputs/'implementation_patch.diff').write_text(''.join(patches))
    names=['src/lerobot/policies/act/configuration_act.py','src/lerobot/policies/act/modeling_act.py',
           'scripts/run_f2_1.py','scripts/run_f2_2.py','scripts/evaluate_f2_2_selected.py',
           'scripts/report_f2_2.py','scripts/autofinish_f2_2.py','scripts/train_f2_2_one.sh']
    hashes={name:hashlib.sha256((BASE/'source'/name).read_bytes()).hexdigest() for name in names}
    (inputs/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
    protected=['gate_config.json','protocol.json','decision.md','normalization.json',
               'split_manifest.json','historical_config.json','cup_prototype_bank_v1.pt']
    for name in ['normalization.json','split_manifest.json','historical_config.json','cup_prototype_bank_v1.pt']:
        assert (inputs/name).read_bytes()==(PARENT/'inputs'/name).read_bytes(), name
    input_hashes={name:hashlib.sha256((inputs/name).read_bytes()).hexdigest() for name in protected}
    (inputs/'input_hashes.json').write_text(json.dumps(input_hashes,indent=2))
    rubric=dict(consistent_episodes_min=8,latency_ratio_max=1.2,nonzero_diagnostic_floor=1e-6,
        arm_promising_rule='no arm MAE degradation (ratio <=1); conservative descriptive interpretation, not significance test',
        context='Reporting interpretation only; does not change frozen primary metric, selection or training',
        frozen_before_selected_evaluation=True,frozen_utc=datetime.now(timezone.utc).isoformat())
    (inputs/'reporting_rubric.json').write_text(json.dumps(rubric,indent=2))
    print(json.dumps(dict(source_hashes=hashes,rubric=rubric),indent=2))


if __name__=='__main__':
    main()
