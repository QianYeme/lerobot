"""Package only new orchestration files and frozen metadata, not training edits."""
import tarfile
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[1]
    result=root/'experiments/force/f2_2_contact_gated_single_frame_load/index1/result'
    parent=root/'experiments/force/f2_1_load_temporal_encoding_ablation/index1/result'
    scripts=['evaluate_f2_2_selected.py','report_f2_2.py','autofinish_f2_2.py',
             'setup_f2_2_autofinish.py','smoke_f2_2_delivery.py','start_f2_2_autofinish.sh']
    with tarfile.open(root/'outputs/f2_2_autofinish.tar','w') as tar:
        for name in scripts:
            tar.add(root/'scripts'/name,arcname='source/scripts/'+name)
        tar.add(result/'gpu_smoke.json',arcname='inputs/gpu_smoke.json')
        tar.add(parent/'historical_metric_reproduction.json',arcname='inputs/historical_metric_reproduction.json')
        tar.add(result.parent/'decision_F2_2_CONTACT_GATED_SINGLE_FRAME_LOAD.md',arcname='inputs/decision.md')


if __name__=='__main__':
    main()
