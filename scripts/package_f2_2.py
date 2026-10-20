"""Build isolated F2.2 overlay; mechanically preserve baseline mixed EOLs."""
import difflib
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    paths = ['src/lerobot/policies/act/configuration_act.py', 'src/lerobot/policies/act/modeling_act.py']
    for name in paths:
        path = ROOT / name
        old = subprocess.check_output(['git', 'show', 'HEAD:' + name], cwd=ROOT).splitlines(keepends=True)
        new = path.read_bytes().splitlines(keepends=True)
        a, b = [v.rstrip(b'\r\n') for v in old], [v.rstrip(b'\r\n') for v in new]
        aligned = [v + b'\n' for v in b]
        for block in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_matching_blocks():
            for offset in range(block.size):
                aligned[block.b + offset] = old[block.a + offset]
        path.write_bytes(b''.join(aligned))
    output = ROOT / 'outputs/f2_2_overlay.tar'
    with tarfile.open(output, 'w') as tar:
        for name in paths + ['scripts/run_f2_2.py', 'scripts/run_f2_1.py']:
            tar.add(ROOT / name, arcname='source/' + name)
        result = ROOT / 'experiments/force/f2_2_contact_gated_single_frame_load/index1/result'
        for name in ['gate_config.json', 'protocol.json']:
            tar.add(result / name, arcname='inputs/' + name)
    print(output)


if __name__ == '__main__':
    main()
