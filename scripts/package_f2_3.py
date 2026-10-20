"""Pack new F2.3 scripts and frozen inputs without mutating prior experiments."""
import tarfile
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[1]
    base=root/'experiments/force/f2_3_global_vs_local_load_regulation/index1'
    with tarfile.open(root/'outputs/f2_3_initial.tar','w') as archive:
        archive.add(root/'scripts/run_f2_3.py',arcname='source/scripts/run_f2_3.py')
        for name in ['residual_config.json','baseline_identity.json']:
            archive.add(base/'result'/name,arcname='inputs/'+name)
        archive.add(base/'decision_F2_3_GLOBAL_VS_LOCAL_LOAD_REGULATION.md',arcname='inputs/decision.md')


if __name__=='__main__':main()
